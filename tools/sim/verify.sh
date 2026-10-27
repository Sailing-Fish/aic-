#!/usr/bin/env bash
# 用指定参数跑一次闭环，自动清理前后进程。
#
#   bash tools/sim/verify.sh                  # 默认 0.44 / 3.30 / 0.35
#   bash tools/sim/verify.sh 0.40 3.0 0.40    # 指定 speed kp damp
#   bash tools/sim/verify.sh --track roundabout 0.40 3.0 0.40
#
# 输出：圈速、4 根黄线的停车时长、出界/丢线次数、里程
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACK="fork_s"
if [[ "${1:-}" == "--track" ]]; then
  TRACK="$2"; shift 2
fi
SPEED="${1:-0.44}"
KP="${2:-3.30}"
DAMP="${3:-0.35}"
YAW="${4:-1.50}"
STOP="${5:-1.00}"

cd "$WS"

cleanup() {
  pkill -f "tools/sim/world_node.py" 2>/dev/null || true
  pkill -f "line_follower_with_detection" 2>/dev/null || true
  pkill -f "line_follower_node" 2>/dev/null || true
  pkill -f "cmd_mux_node" 2>/dev/null || true
  pkill -f "detection_turn_controller" 2>/dev/null || true
}

echo "[1/5] 清理残留"
cleanup
sleep 3
if pgrep -f "tools/sim/world_node.py|line_follower_node" > /dev/null; then
  echo "[ERROR] 仍有残留进程，先手动清理："
  pgrep -af "world_node.py|line_follower_node"
  exit 1
fi

echo "[2/5] 生成参数: speed=$SPEED kp=$KP damp=$DAMP yaw=$YAW stop=$STOP"
source /opt/tros/humble/setup.bash
source install/setup.bash
python3 - "$SPEED" "$KP" "$DAMP" "$YAW" "$STOP" "$TRACK" <<'PY'
import re, sys
speed, kp, damp, yaw, stop, track = sys.argv[1:]
base = f"tools/config/{'roundabout' if track == 'roundabout' else 'fork'}.yaml"
t = open(base).read()
t = t.replace('    target_speed:',
              f'    odom_topic: /sim/odom\n    target_speed:', 1)
t = re.sub(r'(target_speed:\s*)[0-9.]+', rf'\g<1>{speed}', t)
t = re.sub(r'(max_speed:\s*)[0-9.]+', rf'\g<1>{speed}', t)
t = re.sub(r'(^    kp:\s*)[0-9.]+', rf'\g<1>{kp}', t, flags=re.M)
t = re.sub(r'(speed_gain_damping:\s*)[0-9.]+', rf'\g<1>{damp}', t)
t = re.sub(r'(^    max_yaw_rate:\s*)[0-9.]+', rf'\g<1>{yaw}', t, flags=re.M)
t = re.sub(r'(yellow_stop_duration:\s*)[0-9.]+', rf'\g<1>{stop}', t)
t = t.replace('output_topic: /cmd_vel', 'output_topic: /sim/cmd_vel')
open('/tmp/verify_params.yaml', 'w').write(t)
PY
grep -E "odom_topic|target_speed|max_speed|^    kp:|max_yaw_rate|speed_gain_damping|yellow_stop_duration|output_topic" /tmp/verify_params.yaml

echo "[3/5] 启动世界 + 巡线"
python3 -u tools/sim/world_node.py --track "tools/sim/tracks/$TRACK.json" \
  > /tmp/verify_world.log 2>&1 &
WORLD=$!
sleep 4
ros2 launch line_follower line_follower_with_detection.launch.py \
  params_file:=/tmp/verify_params.yaml cmd_vel_topic:=/sim/cmd_vel \
  > /tmp/verify_follow.log 2>&1 &
LAUNCH=$!
sleep 9
grep -a "Cmd mux" /tmp/verify_follow.log | tail -1 || { echo "[ERROR] launch 未就绪"; exit 1; }

echo "[4/5] 发车"
ros2 topic pub -r 10 --times 20 /line_follower/start_cmd std_msgs/msg/Bool \
  "{data: true}" > /dev/null 2>&1 || true
for i in $(seq 1 75); do
  if grep -aq final_stop /tmp/verify_follow.log; then
    echo "      到终点 +${i}s"
    break
  fi
  sleep 1
done

echo "[5/5] 结果"
python3 - <<'PY'
import re
txt = open('/tmp/verify_follow.log', errors='ignore').read()
m = re.search(r'\[(\d+\.\d+)\].*Start signal changed: false -> true', txt)
stops = re.findall(r'\[(\d+\.\d+)\].*yellow stop_begin count=(\d)/4', txt)
ends = re.findall(r'\[(\d+\.\d+)\].*yellow stop_end dur=([0-9.]+)', txt)
fin = re.findall(r'\[(\d+\.\d+)\].*yellow final_stop', txt)
lost = len(re.findall(r'state change from=tracking to=lost', txt))
wtxt = open('/tmp/verify_world.log', errors='ignore').read()
off = re.findall(r'\[sim\] s=[0-9.-]+ off=([0-9.]+)', wtxt)
smax = re.findall(r'\[sim\] s=([0-9.]+)', wtxt)
print(f"  圈速        : {float(fin[-1]) - float(m.group(1)):.2f} s" if (m and fin) else "  圈速        : 未完成")
print(f"  停车次数    : {len(stops)}/4")
print(f"  停车时长    : {[round(float(x),2) for _, x in ends]}")
print(f"  丢线次数    : {lost}")
print(f"  最大出界    : {max(map(float, off)) if off else 0:.2f} s")
print(f"  里程        : {max(map(float, smax)) if smax else 0:.2f} m")
PY

cleanup
echo "[done] 已清理进程"
