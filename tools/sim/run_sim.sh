#!/usr/bin/env bash
# 闭环仿真：sim_world + 现有 line_follower/cmd_mux，不改一行 C++。
#
#   ./tools/sim/run_sim.sh fork_s            # 默认岔路
#   ./tools/sim/run_sim.sh roundabout        # 环岛
#   ./tools/sim/run_sim.sh fork_s --no-start # 只起节点，不发车
#
# 安全：mux 输出被重定向到 /sim/cmd_vel，真底盘的 /cmd_vel 收不到任何指令。
# 注意：不要加 set -u——/opt/tros/humble/setup.bash 会引用未定义的
# AMENT_TRACE_SETUP_FILES，开 nounset 会直接报错退出。
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACK="${1:-fork_s}"
AUTO_START=1
[[ "${2:-}" == "--no-start" ]] && AUTO_START=0

source /opt/tros/humble/setup.bash
source "$WS/install/setup.bash"

# 用真实参数，但把 mux 输出改到仿真话题
SRC_PARAMS="$WS/tools/config/$TRACK.yaml"
[[ -f "$SRC_PARAMS" ]] || SRC_PARAMS="$WS/tools/config/fork.yaml"
TMP_PARAMS="/tmp/sim_${TRACK}_params.yaml"
sed -e 's|output_topic: */cmd_vel$|output_topic: /sim/cmd_vel|' \
    -e 's|output_topic: /cmd_vel|output_topic: /sim/cmd_vel|' \
    "$SRC_PARAMS" > "$TMP_PARAMS"

if pgrep -f "world_node.py" >/dev/null; then
  echo "[warn] 已有 world_node.py 在跑，先停掉它"
  pkill -f "world_node.py" || true
  sleep 1
fi

echo "[1/3] sim_world ($TRACK)"
python3 -u "$WS/tools/sim/world_node.py" \
  --track "$WS/tools/sim/tracks/$TRACK.json" \
  > /tmp/sim_world.log 2>&1 &
WORLD_PID=$!
sleep 3

echo "[2/3] line_follower + cmd_mux"
# cmd_vel_topic 必须显式传：launch 用它覆盖 cmd_mux 的 output_topic，
# 不传就是默认 /cmd_vel，会直接驱动真底盘。
ros2 launch line_follower line_follower_with_detection.launch.py \
  params_file:="$TMP_PARAMS" \
  cmd_vel_topic:=/sim/cmd_vel > /tmp/sim_follow.log 2>&1 &
LAUNCH_PID=$!
sleep 6

# 双保险：确认 mux 真的输出到仿真话题
if ! grep -q "output='/sim/cmd_vel'" /tmp/sim_follow.log 2>/dev/null; then
  echo "[ERROR] cmd_mux 未按预期输出到 /sim/cmd_vel，已中止。"
  echo "        当前日志:"
  grep -a "Cmd mux" /tmp/sim_follow.log | tail -3
  exit 1
fi
echo "      mux output 已确认: /sim/cmd_vel"

if [[ $AUTO_START -eq 1 ]]; then
  echo "[3/3] 发车"
  ros2 topic pub -r 10 /line_follower/start_cmd std_msgs/msg/Bool "{data: true}" \
    > /dev/null 2>&1 &
  START_PID=$!
else
  START_PID=""
  echo "[3/3] 跳过发车（--no-start），手动发："
  echo "  ros2 topic pub -r 10 /line_follower/start_cmd std_msgs/msg/Bool \"{data: true}\""
fi

cleanup() {
  echo
  echo "[stop] 停车并发零速"
  ros2 topic pub --once /line_follower/start_cmd std_msgs/msg/Bool "{data: false}" >/dev/null 2>&1 || true
  ros2 topic pub -r 20 /sim/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" --times 20 >/dev/null 2>&1 || true
  [[ -n "$START_PID" ]] && kill "$START_PID" 2>/dev/null || true
  kill -INT "$LAUNCH_PID" 2>/dev/null || true
  kill "$WORLD_PID" 2>/dev/null || true
  sleep 2
  echo "[done] 日志: /tmp/sim_world.log  /tmp/sim_follow.log"
}
trap cleanup EXIT INT TERM

echo
echo "运行中。Ctrl+C 结束。"
echo "  观察圈速:  ros2 topic echo /sim/gt"
echo "  当前图像:  ros2 run rqt_image_view rqt_image_view  (选 /image_raw)"
echo "  里程/朝向: ros2 topic echo /sim/gt --once"
wait "$LAUNCH_PID"
