#!/bin/bash

echo "=============================="
echo " Start RDK X5 Car System"
echo "=============================="


# Ctrl+C退出处理
cleanup()
{
    echo ""
    echo "Stopping RDK X5 Car System..."

    # 杀整个进程组
    kill -- -$WHEELTEC_PID 2>/dev/null
    kill $CAMERA_PID 2>/dev/null
    kill $YOLO_PID 2>/dev/null

    echo "All processes stopped."
    exit 0
}


trap cleanup SIGINT SIGTERM


# ==============================
# 1. 启动底盘
# ==============================

echo "[1/3] Starting wheeltec robot..."

cd /home/sunrise/ws_rdkx5/src/car_ws

source install/setup.bash

setsid ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py &

WHEELTEC_PID=$!


sleep 5


# ==============================
# 2. 启动相机
# ==============================

echo "[2/3] Starting camera..."

cd /home/sunrise/ws_rdkx5/src/camera

python3 camera.py &

CAMERA_PID=$!


sleep 3


# ==============================
# 3. 启动YOLO
# ==============================

echo "[3/3] Starting YOLOv8..."

cd /home/sunrise/ws_rdkx5/src/yolov8

source /opt/tros/humble/setup.bash

python3 bpu_yolov8_node.py &

YOLO_PID=$!


echo "=============================="
echo " Camera + YOLO + Wheeltec Started"
echo " Press Ctrl+C to stop"
echo "=============================="


wait
