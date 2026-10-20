# RDK X5 小车工作区编译、巡线调参与二次开发说明

> 本文是原 `rdkx5.md`《RDK X5 阿克曼视觉小车运行流程》的配套开发文档。**相机 `camera`、YOLO `yolov8` 和已有底盘功能无需调试**；这里主要介绍 `car_ws`、`line_follow_ws` 在运行前的独立清理编译，以及巡线系统的参数、接口、源码入口。启动顺序和相机/YOLO 的实际运行命令请继续按原 `rdkx5.md` 执行。
>
> **路径注意**：原运行手册使用 `/home/sunrise/ws_rdkx5/src`；下面的命令沿用原手册的**绝对路径**。不要清理错工作区。

## 1. car_ws：运行底盘前，清理旧构建并单独编译

`car_ws` 的位置（原运行手册）：

```text
/home/sunrise/ws_rdkx5/src/car_ws
└── src/turn_on_wheeltec_robot
```

**适用时机**：从旧位置复制了源码、ROS 环境切换、`CMakeCache.txt` 报旧路径，或首次部署新系统时。`build/`、`install/`、`log/` 是该工作区的构建产物，清理后需要重新编译；**不要删除 `src/`、`camera/`、`yolov8/`，也不要在整个 `ws_rdkx5/src` 上执行 `rm -rf`。**

先终止正在运行的底盘 Launch 和依赖该工作区 `install/` 的节点，然后**在一个新的终端**执行：

```bash
# 1) 进入 car_ws
cd /home/sunrise/ws_rdkx5/src/car_ws
pwd -P

# 2) 仅清除 car_ws 自己的历史编译产物
rm -rf -- build install log

# 3) 使用开发板现有的 TROS Humble 基础环境
source /opt/tros/humble/setup.bash

# 4) 确认本工作区能找到 ROS 包后，再独立编译
colcon build 

# 5) 只在编译成功后加载新工作区
source install/setup.bash
ros2 pkg prefix turn_on_wheeltec_robot
```

**重要**：若板上实际不存在 `/opt/tros/humble/setup.bash`，先查看 `ls /opt/tros/humble/setup.bash /opt/ros/humble/setup.bash 2>/dev/null`，以本机实际安装环境为准，不要继续在未加载正确 ROS 环境的终端编译。若遇到串口打不开，这通常不是 `build/` 缓存问题，应按原底盘说明检查设备名、udev 固定串口规则、权限与下位机连接，不要反复删除源码。

底盘启动仍按原运行手册：

```bash
cd /home/sunrise/ws_rdkx5/src/car_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

> 如果你的底盘工作区以前依赖 `/opt/ros/humble` 而不是 TROS，应统一采用**实际能编译该工作区的那套环境**；不要在同一终端混用两套 Humble 的 `setup.bash`。这里采用 TROS 是为了与原手册其他模块的环境保持一致，最终以开发板本机依赖和实际构建结果为准。

## 2. line_follow_ws：刷新 TROS 环境后独立编译

巡线工作区和包的位置（原运行手册）：

```text
/home/sunrise/ws_rdkx5/src/line_follow_ws
└── src/line_follower
    ├── launch/line_follower.launch.py
    ├── launch/line_follower_with_detection.launch.py
    └── config/line_follower_params.yaml
```

以上 Launch、YAML 的包内子目录为常见 ROS 2 包布局，**以实际源码目录和包安装结果为准**，可用 `find src/line_follower -name 'line_follower*.launch.py' -o -name 'line_follower_params.yaml'` 确认。相机和 YOLO 已写好，本节**不重编、不修改**它们。

关闭所有正在运行的巡线节点，在**新终端**执行。尤其是之前复制过 `line_follow_ws`、出现 `ament_cmake` 或 `rosidl_default_generators` 找不到、旧 `CMakeCache.txt` 路径不一致时，建议用下面的完整流程：

```bash
# 1) 进入巡线工作区，确认当前位置
cd /home/sunrise/ws_rdkx5/src/line_follow_ws
pwd -P

# 2) 仅清除 line_follow_ws 的构建产物
rm -rf -- build install log

# 3) 先刷新基础 ROS/TROS 环境；注意是 /opt/tros，不是 /opt/ros
source /opt/tros/humble/setup.bash

# 4) 再单独编译巡线工作区
colcon 

# 5) 编译成功后加载本工作区，确认包和启动文件
source install/setup.bash
ros2 pkg prefix line_follower
```

如果同一个终端之前 `source` 过旧目录的 `install/setup.bash`，仅重新 `source /opt/tros/humble/setup.bash` **不保证清除旧终端中的所有环境叠加**，所以建议直接开新终端执行，避免旧 `AMENT_PREFIX_PATH` / `CMAKE_PREFIX_PATH` 污染编译。不要在 `line_follow_ws` 中 `source` 另一份旧 `line_follow_ws/install/setup.bash`。

重新编译后，两种模式**二选一**启动：

```bash
cd /home/sunrise/ws_rdkx5/src/line_follow_ws
source /opt/tros/humble/setup.bash
source install/setup.bash

# 纯巡线模式
ros2 launch line_follower line_follower.launch.py

# 或：完整巡线 / 数字转向模式（另一个终端，不与上一条同时运行）
# ros2 launch line_follower line_follower_with_detection.launch.py
```

`car_ws`、`line_follow_ws` 要分别在**各自工作区根目录**执行 `colcon build`；不能只在上层 `ws_rdkx5/src` 执行一次就假定两个嵌套工作区都已正确安装。完整启动顺序及 `start_toggle_node`、相机、YOLO 启动命令，继续见原 `rdkx5.md`。

## 3. 巡线系统调参与二次开发（原调参文档整合）

> 适用范围：本次提供的 `line_follower.launch.py`、`line_follower_with_detection.launch.py`、`line_follower_params.yaml` 及对应 C++ 源码。本文是**开发/调试手册**，与已有的整车启动操作说明配套使用。代码版本更新后，先以实际 ROS 参数和话题为准。

#### 3.3.1 先看这里：该改哪个文件？

| 需求 | 优先修改的位置 | 实际执行模块 |
|---|---|---|
| 选择纯巡线或综合功能、切换话题 | `launch/line_follower.launch.py`、`launch/line_follower_with_detection.launch.py` | Launch |
| 车速、转向灵敏度、白线 ROI/阈值、黄线停车、红色避障 | `config/line_follower_params.yaml` → `line_follower_node.ros__parameters` | `line_follower_node`，实现见 `line_follower_node.cpp` |
| YOLO 数字识别后的转向速度、角速度、持续时间、冷却 | 同一 YAML → `detection_turn_controller.ros__parameters` | `detection_turn_controller.cpp` |
| 普通巡线、数字转向、停车之间的抢占 | **先看综合 Launch 中 mux 的显式覆盖**，再看 YAML → `cmd_mux_node` | `cmd_mux_node.cpp` |
| 修改识别/避障算法本身 | `line_follower_node.cpp`、`detection_turn_controller.cpp` | C++ 编译后生效 |
| 相机分辨率转换 | `image_resizer_node.cpp` 的 `input_topic`、`output_topic`、`target_width/height` | 可选节点，两个 Launch 均未启动它 |
| 终端空格开始/暂停 | `start_toggle_node.cpp` 的 `start_topic` | 可选节点，两个 Launch 均未启动它 |
| 旧版合并式数字巡线 | `line_follower_digit_node.cpp` | **未被本次两个 Launch 调用**，不要和当前拆分式控制器混淆 |

**工作区位置**：原运行手册写的是 `/home/sunrise/ws_rdkx5/src/line_follow_ws`，截图中的项目可能位于“文档/RDK X5小车项目/ws_rdkx5/src”；两者可能是不同副本。请先用 `pwd -P` 确认当前使用的项目副本，再用 `find . -name line_follower_params.yaml` 定位配置。以下命令优先沿用原运行手册的绝对路径。上传文件未包含 `package.xml`、`CMakeLists.txt` 和头文件。

#### 3.3.2 两种 Launch 的组成及边界

#### 3.2.1 纯巡线：`line_follower.launch.py`

仅创建一个 `line_follower_node`；默认订阅 `/image_raw`，**直接向 `/cmd_vel` 发布底盘速度**。传入参数包括 `params_file`、`image_topic`、`cmd_vel_topic`，其中 Launch 明确设置的 `cmd_vel_topic=/cmd_vel` 覆盖 YAML 中的 `/cmd_vel_follower`。

注意：**“纯巡线 Launch”不等于自动关闭红色和黄色识别**。它加载同一个 YAML，而本次 YAML 的 `use_yellow_filter=true`、`enable_red_filter=true`、`enable_soft_red_avoidance=true`。如需仅白线跟随，应另外准备一份纯巡线 YAML（关闭黄色/红色功能），通过 `params_file:=...` 指定；不建议直接改动综合模式共用的参数文件。

#### 3.2.2 综合巡线：`line_follower_with_detection.launch.py`

实际只创建三个节点：`line_follower_node`、`detection_turn_controller`、`cmd_mux_node`。黄色停车和红色避障都在 `line_follower_node` 内完成；YOLO 推理节点、相机节点、键盘开始节点**不在该 Launch 内**，需要沿用已有操作说明另行启动。

```text
相机 /image_raw ──> line_follower_node ──> /cmd_vel_follower ─┐
                     │       │                 /cmd_vel_stop ──┤
                     │       ├─> line_follower/centroid ─┐     │
                     │       └─> line_follower/track_state│     │
YOLO /hobot_dnn_detection ───> detection_turn_controller  │     │
                                  └─> /cmd_vel_detection ─┤     │
                                                          └──> cmd_mux_node ──> /cmd_vel ──> 底盘
开始/暂停 line_follower/start_cmd ──> line_follower_node
```

数字转向控制器同时需要 `/hobot_dnn_detection`、`line_follower/centroid`、`line_follower/track_state`；不能只启动检测发布端而不提供质心。相机像素坐标和 YOLO 检测框尺寸也必须对应。

#### 3.3.3 参数到底从哪里生效？

1. **C++ `declare_parameter` 是缺省值**：没有其他配置才使用。
2. **YAML 是本次常规调参入口**：每个顶层节点名必须与实际节点名相匹配，例如 `line_follower_node:`。
3. **Launch 中 `parameters=[params_file, {…}]` 后面的字典优先覆盖 YAML**。本次综合 Launch 把 mux 的 `high_timeout` 显式固定为 **0.5 s**，尽管 YAML 写的是 **0.12 s**；`stop_timeout` 同样在 Launch 固定为 0.5 s。修改 YAML 中的 `high_timeout` 不会改变本次综合 Launch 的实际值，应改 Launch 或移除对应覆盖项。
4. C++ 中有的参数仅在构造时读取，有的在图像回调内反复 `get_parameter`。**修改 YAML 后重启节点最可靠**；`ros2 param set` 是否影响当前运行取决于源码是否再次读取该参数，不要默认所有参数都能热更新。
5. 调试实际生效值：`ros2 param get /line_follower_node kp`、`ros2 param get /detection_turn_controller turn_duration`、`ros2 param get /cmd_mux_node high_timeout`，不要只看文件内容。

#### 3.3.4 巡线和视觉调参（`line_follower_node`）

以下数值为** YAML 的配置值**。

| 参数 | 当前值 | 作用 / 调整观察点 |
|---|---:|---|
| `target_speed` | 0.3 m/s | 期望前进速度；先低速校准转向再提高 |
| `max_speed` | 0.4 m/s | 正常速度上限 |
| `min_speed` | 0.0 m/s | 速度下限，具体停车和丢线行为还受状态逻辑控制 |
| `max_yaw_rate` | 1.5 rad/s | 最大角速度 |
| `kp` | 2.45 | 白线横向误差的转向比例增益；振荡时应结合速度一起检查 |
| `roi_top_ratio` / `roi_bottom_ratio` | 0.65 / 0.20 | 巡线处理区域边界；改动后看 debug 图是否覆盖所需车道 |
| `binary_threshold` | 160 | 二值化阈值，需与白色筛选联合调试 |
| `use_white_filter` / `white_s_max` / `white_v_min` | true / 45 / 80 | 白色 HSV 筛选；亮度变化下检查误检与漏检 |
| `min_contour_area` | 2500 | 过滤过小轮廓；调大可能漏远处细线 |
| `enable_continuous_curve_assist` | true | 连续弯辅助开关 |
| `reverse_curve_min_error` / `reverse_curve_threshold` | 0.12 / 0.35 | 反向连续弯辅助判定相关阈值 |
| `reverse_curve_speed` / `sharp_curve_speed` | 0.16 / 0.20 m/s | 连续弯、急弯对应速度 |
| `line_lost_stop_time` / `line_lost_forward_speed` | 0.8 s / 0.3 m/s | 丢线状态判定及后续低速前进配置；本版源码注明丢线不停车 |
| `publish_debug_image` | true | 发布 `line_follower/debug_image` |

调试顺序：固定相机高度与角度 → 用 debug 图观察 ROI、白线掩膜和质心 → 调阈值与轮廓面积 → 低速调 `kp` → 调速度与连续弯参数。若底盘转向方向错误，应先检查底盘坐标约定与误差符号，而不是单纯提高增益。

#### 3.4.1 黄色停车

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `use_yellow_filter` | true | 启用黄色识别 |
| `yellow_h_low` / `yellow_h_high` | 20 / 40 | HSV 色相窗口 |
| `yellow_s_min` / `yellow_v_min` | 80 / 120 | 饱和度、亮度下限 |
| `yellow_area_ratio_threshold` | 0.02 | 黄色像素面积比例触发阈值 |
| `yellow_stop_duration` | 1.0 s | 非最终黄线临时停车时长 |
| `yellow_permanent_count` | 4 | 达到第 4 次不同黄线触发永久停车 |
| `yellow_clear_frames` | 10 | 连续看不到黄色多少帧后解除当前黄线锁存，允许计下一条 |

代码中的黄线检测计数采用锁存与清除帧数，避免同一条黄色线在连续画面中被反复计数。永久停车后，`start_cmd` 从 `false` 重新变为 `true` 才清除永久停车并重置下一圈黄线计数；普通中途暂停/继续不清零。

#### 3.4.2 红色避障：本版使用软避障

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `enable_red_filter` / `enable_soft_red_avoidance` | true / true | 启用红色识别和动态软避障 |
| `red_h_low_1`~`red_h_high_1` / `red_h_low_2`~`red_h_high_2` | 0–10 / 160–180 | 两段红色 HSV 色相窗口 |
| `red_s_min` / `red_v_min` | 90 / 70 | 红色饱和度、亮度下限 |
| `red_line_aspect_ratio_min` / `red_line_min_width_ratio` | 3.0 / 0.30 | 区分横向红线与红锥的形状、宽度条件 |
| `red_cone_area_ratio_threshold` | 0.0006 | 单个红锥轮廓面积比例门限 |
| `red_near_ratio` / `red_very_close_ratio` | 0.38 / 0.72 | 基于目标框底部位置的接近程度阈值 |
| `red_corridor_half_width_ratio` | 0.35 | 目标是否进入行驶走廊的判断范围 |
| `red_soft_yaw` / `red_repulsion_gain` | 0.15 / 1.50 | 基础避让角速度与随风险增强的斥力 |
| `red_soft_speed` / `red_min_forward_speed` | 0.25 / 0.15 m/s | 软避障降速与最低前进速度 |
| `red_line_fallback_direction` | -1 | 左右空间接近时，红色横线的固定绕行方向；源码注释中 -1 为右绕 |

特别注意：在本版 YAML 中开启软避障时，`red_turn_yaw`、`red_turn_linear`、`red_turn_duration` 是保留的旧固定时间避障参数，**不是软避障主要调参入口**。红色目标的形状分类、风险计算及叠加转向在 `line_follower_node.cpp`，调红色误触发时先检查颜色掩膜、面积阈值和走廊判断。

#### 3.3.5 YOLO 数字识别与转向（`detection_turn_controller`）

YOLO 模型和推理过程由外部识别节点负责；本 C++ 节点接收 `ai_msgs/msg/PerceptionTargets`，从 ROI 的类别 `type` 和 `confidence` 中识别数字 1–8（同时兼容 `D1`–`D8` 形式）。在候选中优先数字较大的类别，同数字再比较置信度；此外还要与巡线质心在 Y 方向对齐。本节参数**只调检测结果到运动指令的转换**，不负责改 YOLO 模型本身。

| 参数 | 当前 YAML 值 | 作用 |
|---|---:|---|
| `confidence_threshold` | 0.1 | 此控制器接收检测框的置信度门限，不等于 YOLO 推理端的置信度 |
| `det_frame_width` / `det_frame_height` | 640 / 480 | 解释检测框像素坐标的画面尺寸，必须匹配实际检测结果 |
| `frame_window` | 2 | 采集候选的帧数窗口 |
| `y_align_tolerance` | 0.01 | 数字中心与白线质心归一化 Y 差的容许值 |
| `centroid_timeout` | 0.5 s | 白线质心新鲜度门限；过期不触发数字转向 |
| `yaw_gain` / `max_yaw_rate` | 1.6 / 1.3 rad/s | X 方向位置差转换为角速度并限幅 |
| `turn_direction_sign` | -1.0 | 转向符号；反向时先核对物理坐标与此参数 |
| `turn_linear` / `turn_duration` | 0.20 m/s / 0.38 s | 数字转向时的直行速度和持续时间 |
| `turn_cooldown` | 0.8 s | 每次转向后的识别冷却时间 |

控制器在收到巡线状态 `yellow_line_detected` 时会清空当前数字识别会话并解除冷却，便于黄线之后重新识别。若数字可视化有框但车不转，应依次检查：检测话题的真实类型和 QoS、质心是否持续发布、检测框坐标尺寸、Y 对齐阈值、转向输出和 mux 抢占。

#### 3.3.6 速度仲裁（`cmd_mux_node`）与容易忽略的覆盖

当前逻辑优先级：**`/cmd_vel_stop`（最高） > `/cmd_vel_detection`（高） > `/cmd_vel_follower`（低）**，最后输出 `/cmd_vel`。`cmd_mux_node.cpp` 采用收到最近指令后的超时窗口判断是否继续抢占；高优先级指令持续发布时，普通巡线不会接管。

| 参数 | YAML 值 | 综合 Launch 实际值 | 用途 |
|---|---:|---:|---|
| `stop_timeout` | 0.5 s | 0.5 s | 停车指令保持窗口 |
| `high_timeout` | 0.12 s | **0.5 s** | 数字转向速度指令抢占窗口 |
| `stop_topic` / `high_topic` / `low_topic` | `/cmd_vel_stop` / `/cmd_vel_detection` / `/cmd_vel_follower` | 同左 | 三路输入 |
| `output_topic` | `/cmd_vel` | `/cmd_vel` | 底盘最终速度 |

`session_cooldown: 2.0` 出现在所附 YAML，但**本次 `cmd_mux_node.cpp` 没有声明或使用这一参数**，不能把它视为生效的 mux 冷却设置。若数字转完迟迟不恢复巡线，优先查看 Launch 中的 `high_timeout=0.5`、`/cmd_vel_detection` 是否仍在发布以及 `/cmd_vel_stop` 的保留窗口。

#### 3.3.7 ROS 2 接口话题速查

| 话题（默认） | 类型 | 发布方 → 订阅方 | 排查目的 |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/msg/Image` | 外部相机 → `line_follower_node`；外部 YOLO 输入视其启动配置而定 | 画面是否到达、帧率和分辨率 |
| `/hobot_dnn_detection`（代码默认配置写作 `hobot_dnn_detection`） | `ai_msgs/msg/PerceptionTargets` | 外部 YOLO → `detection_turn_controller` | 数字框与分类结果；注意相对话题受命名空间影响 |
| `/line_follower/start_cmd` | `std_msgs/msg/Bool` | 外部开关/`start_toggle_node` → `line_follower_node` | `true` 开始、`false` 暂停 |
| `/line_follower/track_state` | `std_msgs/msg/String` | `line_follower_node` → `detection_turn_controller` | `tracking`、`lost`、`yellow_line_detected` 等状态 |
| `/line_follower/centroid` | `geometry_msgs/msg/Point` | `line_follower_node` → `detection_turn_controller` | `x,y` 均为整图归一化质心坐标 |
| `/line_follower/debug_image` | `sensor_msgs/msg/Image` | `line_follower_node` → 调试工具 | 查看 ROI、二值化/检测叠加 |
| `/cmd_vel_follower` | `geometry_msgs/msg/Twist` | `line_follower_node` → `cmd_mux_node` | 普通巡线控制输出（综合模式） |
| `/cmd_vel_detection` | `geometry_msgs/msg/Twist` | `detection_turn_controller` → `cmd_mux_node` | 数字转向控制输出 |
| `/cmd_vel_stop` | `geometry_msgs/msg/Twist` | `line_follower_node` → `cmd_mux_node` | 黄线/永久停车等零速度请求 |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 纯巡线节点**或**综合模式 mux → 底盘 | 最终执行命令，禁止两个模式同时向此话题竞争发布 |

相对话题最终名称需通过 `ros2 node info` 实测；上表按当前无额外命名空间的 Launch 解释。**不要同时运行两个 Launch**，否则纯巡线与 mux 可能同时向 `/cmd_vel` 发布速度，行为不可预期。

#### 3.3.8 常用调试命令

```bash
# 每个新终端先加载 ROS 和工作区（根据实际工作区调整）
source /opt/tros/humble/setup.bash
# source ~/ws_rdkx5/src/line_follow_ws/install/setup.bash

# 先确认节点和实际话题名称/类型
ros2 node list
ros2 topic list -t
ros2 node info /line_follower_node
ros2 node info /detection_turn_controller
ros2 node info /cmd_mux_node

# 图像、检测与质心频率
ros2 topic hz /image_raw
ros2 topic hz /hobot_dnn_detection
ros2 topic echo /line_follower/centroid
ros2 topic echo /line_follower/track_state

# 分路查看速度，不要只看最终 /cmd_vel
ros2 topic echo /cmd_vel_follower
ros2 topic echo /cmd_vel_detection
ros2 topic echo /cmd_vel_stop
ros2 topic echo /cmd_vel

# 查看当前实际参数（检测 Launch 覆盖最有用）
ros2 param get /line_follower_node target_speed
ros2 param get /line_follower_node kp
ros2 param get /detection_turn_controller turn_duration
ros2 param get /cmd_mux_node high_timeout

# 外部节点未启动时可用一条 Bool 消息验证开始接口
ros2 topic pub --once /line_follower/start_cmd std_msgs/msg/Bool '{data: true}'
# 暂停
ros2 topic pub --once /line_follower/start_cmd std_msgs/msg/Bool '{data: false}'
```

注意：`start_toggle_node.cpp` 每 100 ms 发布一次当前状态，终端**空格**切换启动/暂停；若它在运行，其他终端的单次 `ros2 topic pub --once` 会很快被持续状态覆盖。排查时只保留一个启动状态发布源。查看图像可用已有 `rqt_image_view` 工具选择 `/line_follower/debug_image`，如果为空先检查 `publish_debug_image=true` 和相机是否收到画面。

#### 3.3.9 修改后如何编译、验证

首次部署、移动目录或遇到旧缓存冲突时，优先按照本文第 1、2 节清理并分别重新编译。日常只改 C++ 时可在确认工作区根目录后增量编译；不要在重复嵌套的工作区内误编译：

```bash
source /opt/tros/humble/setup.bash
cd ~/ws_rdkx5/src/line_follow_ws
colcon list | grep line_follower
colcon build --packages-select line_follower --symlink-install
source install/setup.bash

# 对应模式二选一启动（相机、YOLO 与底盘沿用原操作手册）
ros2 launch line_follower line_follower.launch.py
# 或
ros2 launch line_follower line_follower_with_detection.launch.py
```

改 YAML 或 Launch 后，重启对应 Launch 并用 `ros2 param get` 校验。改 C++ 后重新编译并重启节点。若 C++ 调参新增了成员变量，还可能需要修改未随此次上传的 `include/line_follower/line_follower_node.hpp`；`main.cpp` 只是构造并 spin 巡线节点的入口，通常不在这里调算法参数。若改动了新节点或可执行文件名称，还应同步检查未上传的 `CMakeLists.txt` / `package.xml` 安装与依赖配置。

#### 3.3.10 按故障定位入口

| 现象 | 首先检查 |
|---|---|
| 相机有画面但巡线无速度 | `image_topic`、`start_cmd`、`line_follower/debug_image`、`/cmd_vel_follower`；纯模式看 `/cmd_vel` |
| 白线频繁丢失 / 走 S 弯 | ROI、白色 HSV、轮廓面积、`kp`、速度、debug 图 |
| 黄线计数不对 | `yellow_area_ratio_threshold`、`yellow_clear_frames`、是否同一条黄线重复跨越；查看 `track_state` 和日志 |
| 红色物体误避让 | 红色 HSV、红线/红锥分类、单轮廓面积、行驶走廊与 debug 图 |
| YOLO 有框但不转向 | `/hobot_dnn_detection` 类型与数据、质心新鲜度、Y 对齐、`/cmd_vel_detection` |
| 已转向却迟迟不回巡线 | `/cmd_vel_detection` 是否继续发、Launch 的 `high_timeout=0.5`、`/cmd_vel_stop` |
| 停车后无法重新开跑 | `start_cmd` 发布源、永久停车状态；永久停车后需经历 `false → true` 新一圈复位 |
| `/cmd_vel` 有值但车不动 | 底盘启动/串口、底盘订阅的话题和底盘使能；先区分感知控制与驱动故障 |

#### 3.3.11 本次文件的开发注意事项

- 本次 `line_follower_digit_node.cpp` 属于另一种“巡线 + 数字”合并实现，**并非**综合 Launch 使用的 `detection_turn_controller.cpp`；其 `min_digit_confidence` 等参数不要直接填入当前 `detection_turn_controller` 配置。
- `image_resizer_node.cpp` 默认输入 `/ascamera/camera_publisher/rgb/image0`，输出相对话题 `image_resized`，默认 320×240；本次两种 Launch 均直接使用 `/image_raw`，不会自动执行缩放。
- `detection_listener.cpp` 是只订阅并打印检测数量/FPS 的辅助节点，不负责转向，两个 Launch 未调用。
- 当前 YAML 中有些附加参数在 `line_follower_node.cpp` 的图像回调里读取，有些构造时读取，实际热更新行为并不统一；以重启验证为准。
- 所附文件中部分 C++ 中文注释有编码异常，阅读或修改前统一检查 UTF-8；不要根据乱码注释推断参数作用，应对照实际代码。
- 本文仅确认此次上传源码范围内的接口；外部相机、YOLO 模型文件、底盘串口/驱动配置及完整启动脚本应继续参考之前的整车操作文档。
