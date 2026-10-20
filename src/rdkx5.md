# RDK X5 阿克曼视觉小车运行流程

## 1. 项目目录

```
/home/sunrise/ws_rdkx5/src

├── camera
│   └── camera.py
├── yolov8
│   ├── bpu_yolov8_node.py
│   └── mod1
├── car_ws
│   └── src/turn_on_wheeltec_robot
└── line_follow_ws
    └── src/line_follower
```

---

## 2. 启动底盘

进入 wheeltec 工作空间：

```bash
cd /home/sunrise/ws_rdkx5/src/car_ws
source install/setup.bash
```

启动：

```bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

---

## 3. 启动摄像头

进入 camera 目录：

```bash
cd /home/sunrise/ws_rdkx5/src/camera
python3 camera.py
```

输出：

```
/image_raw
```

说明：camera 节点无需 source 环境。

---

## 4. 启动 YOLOv8

进入 yolov8 目录：

```bash
cd /home/sunrise/ws_rdkx5/src/yolov8
source /opt/tros/humble/setup.bash
python3 bpu_yolov8_node.py
```

输出：

```
hobot_dnn_detection
```

---

## 5. 普通巡线模式

功能：
- 白线巡线
- 不包含 YOLO 数字识别

启动：

```bash
cd /home/sunrise/ws_rdkx5/src/line_follow_ws

source /opt/tros/humble/setup.bash
source install/setup.bash

ros2 launch line_follower line_follower.launch.py
```

开启巡线：

```bash
ros2 topic pub /line_follower/start_cmd std_msgs/msg/Bool "data: true" --time 3
```

---

## 6. 完整巡线模式（比赛模式）

功能：
- 白线巡线
- YOLOv8 数字识别
- 数字转向控制

启动顺序：

### 1）摄像头

```bash
cd /home/sunrise/ws_rdkx5/src/camera
python3 camera.py
```

### 2）YOLOv8

```bash
cd /home/sunrise/ws_rdkx5/src/yolov8

source /opt/tros/humble/setup.bash

python3 bpu_yolov8_node.py
```

### 3）巡线控制

```bash
cd /home/sunrise/ws_rdkx5/src/line_follow_ws

source /opt/tros/humble/setup.bash
source install/setup.bash

ros2 launch line_follower line_follower_with_detection.launch.py
```

启动控制：

```bash
ros2 run line_follower start_toggle_node
```

---

## 7. 完整比赛启动顺序

```
1. turn_on_wheeltec_robot
2. camera.py
3. bpu_yolov8_node.py
4. line_follower_with_detection.launch.py
5. start_toggle_node
```

---

## 8. 调试

### 键盘控制

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 图像话题查看

```bash
ros2 run rqt_image_raw rqt_image_raw
```

用于查看相机图像话题和 YOLOv8 识别相关图像话题，检查：

- 相机是否正常发布图像
- YOLOv8 是否正常运行
- 识别结果是否正确