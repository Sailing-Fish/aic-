# RDK X5 阿克曼视觉小车运行手册

> 本文按 **2026-10-20 在本机的实际测试结果** 重写，替换旧版中过时的路径与命令。
> 标注说明：✅ 已在本机实测通过　⬜ 仅核对代码/配置，本次未实跑　⚠️ 已知问题或限制

---

## 0. 本机实际情况（先看这一节）

| 项目 | 实际值 |
|---|---|
| 工作区路径 | `/home/sunrise/ws_rdkx5_1`（旧文档写的 `/home/sunrise/ws_rdkx5` 不存在） |
| 系统 | Ubuntu 22.04.5 LTS |
| ROS 基础环境 | `/opt/tros/humble`（编译必须用它） |
| 底盘串口 | `/dev/wheeltec_controller` → `ttyACM0` |
| 底盘型号 | `mini_akm`（阿克曼转向） |
| 串口参数 | 115200 8N1 |
| 键盘遥控 | 已安装 `teleop_twist_keyboard` |
| 版本管理 | 已初始化 git（`main` 分支），构建产物已被忽略 |

### 目录结构

```
/home/sunrise/ws_rdkx5_1
├── src
│   ├── car_ws/src
│   │   ├── turn_on_wheeltec_robot   底盘驱动、EKF、URDF 启动
│   │   ├── serial_ros2              串口库
│   │   ├── wheeltec_robot_msg       自定义消息
│   │   └── wheeltec_robot_urdf      车体模型
│   ├── line_follow_ws/src/line_follower   巡线与数字识别
│   ├── camera/camera.py              USB 相机发布节点
│   ├── yolov8/                       BPU YOLOv8 检测节点与模型
│   └── scripts/                      一键启动脚本（路径为旧路径，需自行修改）
├── build / install / log             colcon 构建产物（不入库）
└── .gitignore
```

根目录 `colcon` 只能识别到 5 个包：`line_follower`、`serial`、`turn_on_wheeltec_robot`、`wheeltec_robot_msg`、`wheeltec_robot_urdf`。两个子工作区的 `install/` 都放了 `COLCON_IGNORE`，不会被重复编译。

---

## 1. 编译

```bash
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash
colcon build
```

要点：

- **必须在根目录 `colcon build`**，不能只在 `src/car_ws` 或 `src/line_follow_ws` 里单独编。
- **必须用 `/opt/tros/humble`**。用 `/opt/ros/humble` 编会在 `line_follower` 处报找不到 `ai_msgs`。
- 只改了一个包时可加速：`colcon build --packages-select turn_on_wheeltec_robot`
- 每开一个新终端，都要重新 `source` 两个环境，缺一不可：

```bash
source /opt/tros/humble/setup.bash
source /home/sunrise/ws_rdkx5_1/install/setup.bash
```

### 常见编译错误

```
The build time path "/home/sunrise/ws_rdkx5/install/serial" doesn't exist
```

原因：`build/` 是从旧路径 `/home/sunrise/ws_rdkx5` 拷贝过来的缓存。处理：

```bash
cd /home/sunrise/ws_rdkx5_1
rm -rf build install log
source /opt/tros/humble/setup.bash
colcon build
```

---

## 2. 启动底盘 ✅

### 2.1 串口准备（已固化，正常无需操作）

底盘是 CH9102 转串口（`1a86:55d4`，序列号 `5CF0015366`），系统里的设备名是 `/dev/ttyACM0`，而节点写死要 `/dev/wheeltec_controller`。已通过 udev 规则固定：

```bash
# /etc/udev/rules.d/wheeltec_controller.rules
KERNEL=="ttyACM*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", \
  ATTRS{serial}=="5CF0015366", MODE="0777", GROUP="dialout", SYMLINK+="wheeltec_controller"
```

验证：

```bash
ls -l /dev/wheeltec_controller     # 应指向 ttyACM0
```

> 该下位机只有在 **DTR/RTS 拉高** 时才持续上报数据。这一动作已写进 `wheeltec_robot.cpp` 的串口初始化里，无需手动处理；换机器/重刷固件后若发现 `/odom` 没有数据，先查这里。

### 2.2 启动

```bash
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

正常启动日志里应出现：

```
[wheeltec_robot]: wheeltec_robot serial port opened
[imu_filter_madgwick]: First IMU message received.
```

启动的节点：`wheeltec_robot`、`ekf_filter_node`、`imu_filter_madgwick`、`robot_state_publisher_mini_akm`、`joint_state_publisher` 及若干静态 TF。

### 2.3 自检

另开终端：

```bash
source /opt/tros/humble/setup.bash
source /home/sunrise/ws_rdkx5_1/install/setup.bash

ros2 topic hz /odom                       # 正常约 20~23 Hz，无输出就是串口/下位机问题
ros2 topic echo /PowerVoltage --once      # 电池电压，满电约 12.6V
```

### 2.4 话题一览

| 话题 | 类型 | 说明 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 底盘速度输入，`linear.x` 前进 m/s，`angular.z` 转向 rad/s |
| `/odom` | `nav_msgs/Odometry` | 底盘原始里程计，`odom_combined` → `base_footprint` |
| `/odom_combined` | `nav_msgs/Odometry` | EKF 融合后的里程计 |
| `/imu/data_raw` | `sensor_msgs/Imu` | 下位机原始 IMU |
| `/imu/data` | `sensor_msgs/Imu` | Madgwick 滤波后 IMU |
| `/PowerVoltage` | `std_msgs/Float32` | 电池电压 |

---

## 3. 键盘遥控 ✅

终端 2：

```bash
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.5
```

默认速度是 `0.5 m/s` / `1.0 rad/s`，对这台车偏快，建议按上面的参数压到 `0.1` / `0.5`。

```
Moving around:
   u    i    o
   j    k    l
   m    ,    .
```

| 键 | 动作 |
|---|---|
| `i` / `,` | 前进 / 后退 |
| `j` / `l` | 原地左转 / 右转 |
| `u` / `o` | 前进+左转 / 前进+右转 |
| `m` / `.` | 后退+右转 / 后退+左转 |
| `k` 或其他任意键 | **停车** |
| `q` / `z` | 线速度与角速度同时 ±10% |
| `w` / `x` | 只调线速度 ±10% |
| `e` / `c` | 只调角速度 ±10% |
| `Ctrl+C` | 退出 |

### ⚠️ 停车必须按 `k`

下位机固件没有超时保护，**会一直保持最后一条速度指令**：

- 直接 `Ctrl+C` 退出遥控，车会继续按最后速度跑
- 用 `ros2 topic pub --once` 发一条指令，同样会一直跑
- 唯一可靠的停车方式是显式下发零速

```bash
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}" --times 20
```

> 想验证底盘能不能动，用低速短距离指令，例如 0.05 m/s，随时准备发零速。

---

## 4. 启动摄像头 ⬜

```bash
cd /home/sunrise/ws_rdkx5_1/src/camera
python3 camera.py
```

参数（可在源码中改）：

| 参数 | 默认值 |
|---|---|
| `video_device` | `/dev/video0` |
| `image_width` / `image_height` | 640 × 480 |
| `framerate` | 30.0 |
| `topic_name` | `/image_raw` |

依赖 `cv_bridge` 和 `cv2`，需在 source 过 ROS 环境的终端里运行。验证：

```bash
ros2 topic hz /image_raw
```

---

## 5. 启动 YOLOv8 检测 ⬜

```bash
cd /home/sunrise/ws_rdkx5_1/src/yolov8
source /opt/tros/humble/setup.bash
python3 bpu_yolov8_node.py
```

- 模型：`mod1/best_modified_yolov8.bin`
- 输出话题：`hobot_dnn_detection`
- 依赖 TROS 的 BPU 推理库，**必须**用 `/opt/tros/humble` 环境

---

## 6. 巡线

### 6.1 纯巡线模式 ⬜

功能：只做白线巡线，不做数字识别。

```bash
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch line_follower line_follower.launch.py
```

启动巡线（`--times` 不是旧文档里的 `--time`）：

```bash
ros2 topic pub -r 10 line_follower/start_cmd std_msgs/msg/Bool "{data: true}" --times 20
```

停止巡线：

```bash
ros2 topic pub -r 10 line_follower/start_cmd std_msgs/msg/Bool "{data: false}" --times 20
```

数据流：订阅 `/image_raw` → 输出 `/cmd_vel`。

> **本次修正**：原 `line_follower.launch.py` 里参数顺序是 `[{...}, params_file]`，YAML 里的 `cmd_vel_topic: /cmd_vel_follower` 会覆盖 launch 参数，导致速度发到 `/cmd_vel_follower`，而底盘只订阅 `/cmd_vel`，**车根本不会动**。现已改为 `[params_file, {...}]`，实测 `/cmd_vel` 出现发布者。

### 6.2 完整巡线 + 数字识别 ⬜

启动顺序（三个终端）：

```bash
# 终端 1：底盘
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash && source install/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# 终端 2：摄像头
cd /home/sunrise/ws_rdkx5_1/src/camera && python3 camera.py

# 终端 3：YOLOv8
cd /home/sunrise/ws_rdkx5_1/src/yolov8
source /opt/tros/humble/setup.bash
python3 bpu_yolov8_node.py

# 终端 4：巡线 + 检测 + 速度仲裁
cd /home/sunrise/ws_rdkx5_1
source /opt/tros/humble/setup.bash && source install/setup.bash
ros2 launch line_follower line_follower_with_detection.launch.py
```

启动/暂停巡线（空格切换 true/false，节点以 10Hz 持续发布）：

```bash
ros2 run line_follower start_toggle_node
```

速度仲裁逻辑（`cmd_mux_node`）：

```
line_follower_node        → /cmd_vel_follower    优先级低
detection_turn_controller → /cmd_vel_detection   优先级高（超时 0.5s）
stop 输出                 → /cmd_vel_stop        最高优先级（超时 0.5s）
                    ↓ 合并
                  /cmd_vel → 底盘
```

### 6.3 完整比赛启动顺序

```
1. turn_on_wheeltec_robot.launch.py   （底盘）
2. camera.py                          （相机）
3. bpu_yolov8_node.py                 （检测）
4. line_follower_with_detection.launch.py （巡线+仲裁）
5. start_toggle_node                  （空格开始/暂停）
```

---

## 7. 调试

| 目的 | 命令 |
|---|---|
| 看节点 | `ros2 node list` |
| 看话题 | `ros2 topic list` |
| 看频率 | `ros2 topic hz /odom` |
| 看单条消息 | `ros2 topic echo /odom --once` |
| 看谁在收发 | `ros2 topic info /cmd_vel` |
| 看图像 | `ros2 run rqt_image_view rqt_image_view` |
| 看 TF | `ros2 run tf2_tools view_frames` |

> 旧文档写的 `rqt_image_raw` 本机不存在，正确包名是 `rqt_image_view`。

---

## 8. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动报 `can not open serial port` | 设备名不存在或权限不足 | `ls -l /dev/wheeltec_controller`；确认 udev 规则；确认用户在 `dialout` 组 |
| `/odom` 没有数据 | 下位机未上电 / DTR 未拉高 / 串口被占用 | `fuser -v /dev/ttyACM0` 查占用；确认底盘电池开关已开 |
| 键盘遥控发了但车不动 | 话题不匹配 | `ros2 topic info /cmd_vel` 应能看到 1 个发布者和 1 个订阅者 |
| 车停不下来 | 下位机无超时保护 | 显式发全零 `Twist`（见 3 节） |
| 编译报旧路径 `ws_rdkx5` | 构建缓存来自旧路径 | `rm -rf build install log` 后重编 |
| 编译 `line_follower` 报 `ai_msgs` | 用了 `/opt/ros/humble` | 改用 `source /opt/tros/humble/setup.bash` |
| 电池电压低于 11.5V | 电量不足，可能影响电机 | 先充电再跑 |

---

## 9. 已知限制

1. **下位机没有超时保护**。固件无源码，无法在底层加看门狗；主机侧曾尝试加"无指令自动停车"，因影响手动操作已撤掉。**任何时候离开前都要确认车已停**。
2. **电量要求**。实测静态电压约 11.9V，属偏低状态（3S 满电 12.6V），跑之前建议充满。
3. **巡线/相机/YOLO 未在本次实测**。本文第 4~6 节的命令依据代码与配置核对，底盘链路（第 2、3 节）已实测通过。
4. **`src/scripts/start_camera_yolo_car.sh` 用的是旧路径** `/home/sunrise/ws_rdkx5`，直接执行会失败，需要先把里面的路径改成 `/home/sunrise/ws_rdkx5_1`。

---

## 10. 速查

```bash
# 环境（每个新终端都要）
source /opt/tros/humble/setup.bash
source /home/sunrise/ws_rdkx5_1/install/setup.bash

# 编译
cd /home/sunrise/ws_rdkx5_1 && colcon build

# 底盘
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# 遥控
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.5

# 停车（记得！）
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --times 20

# 自检
ros2 topic hz /odom
ros2 topic echo /PowerVoltage --once
```
