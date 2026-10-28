#!/usr/bin/env bash
# RDK X5 智能车 · 一键启动与守护
#
#   ./tools/start.sh                     # 全部拉起（5 个窗口 + 面板 + 兜底 + DEC 汇总）
#   ./tools/start.sh --params roundabout # 用环岛参数
#   ./tools/start.sh --no-panel          # 不起面板
#   ./tools/start.sh --headless          # 不开窗口（无 X11 时用，也能做自检）
#   ./tools/start.sh --stop follow       # 单独停掉某个进程且不自动重启
#   ./tools/start.sh --status            # 打印当前状态
#
# 设计见 tools/DESIGN.md。核心约束：
#   1) 必须在 MobaXterm 的 SSH 终端直接运行，不能套 screen（会丢 DISPLAY）
#   2) 只有退出码非 0 且非 129 才自动重启（0=正常退出，129=关窗口）

set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$WS/tools"
STATE="/tmp/rdkx5"
LOGROOT="$TOOLS/logs"
KEEP_RUNS=10
MAX_RESTARTS=3
FOLLOW_GRACE=25          # ④ launch 启动后多少秒内不检查节点数
NODE_CHECK_EVERY=5

# ---------------------------------------------------------------- 参数解析
PARAMS="fork"
NO_PANEL=0
HEADLESS=0
STOP_TARGET=""
SHOW_STATUS=0
PORT_HINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --params)   PARAMS="${2:-}"; shift 2 ;;
    --no-panel) NO_PANEL=1; shift ;;
    --headless) HEADLESS=1; shift ;;
    --stop)     STOP_TARGET="${2:-}"; shift 2 ;;
    --status)   SHOW_STATUS=1; shift ;;
    --port)     PORT_HINT="${2:-}"; shift 2 ;;
    -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

PARAMS_FILE="$TOOLS/config/$PARAMS.yaml"

# ---------------------------------------------------------------- 小工具
log()  { printf '[%s] %b\n' "$(date '+%H:%M:%S')" "$*"; }
die()  { printf '\033[31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

read_file() { [[ -f "$1" ]] && cat "$1" 2>/dev/null; }
proc_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

# ---------------------------------------------------------------- --status
if [[ $SHOW_STATUS -eq 1 ]]; then
  if [[ -f "$STATE/status.json" ]]; then
    python3 -m json.tool "$STATE/status.json"
  else
    echo "没有运行中的实例（$STATE/status.json 不存在）"
  fi
  exit 0
fi

# ---------------------------------------------------------------- --stop
if [[ -n "$STOP_TARGET" ]]; then
  pidf="$STATE/$STOP_TARGET.pid"
  [[ -f "$pidf" ]] || die "找不到 $STOP_TARGET 的 pid 文件：$pidf"
  pid="$(read_file "$pidf")"
  if proc_alive "$pid"; then
    # 打抑制标记，守护看到后不会重启
    touch "$STATE/$STOP_TARGET.suppress"
    kill -TERM "$pid" 2>/dev/null
    log "已请求停止 $STOP_TARGET (pid=$pid)，守护不会重启它"
  else
    log "$STOP_TARGET 已不在运行"
  fi
  exit 0
fi

# ---------------------------------------------------------------- 前置检查
[[ -f "$PARAMS_FILE" ]] || die "参数文件不存在: $PARAMS_FILE（可选: fork / roundabout）"
[[ -f "$WS/install/setup.bash" ]] || die "工作区未编译: $WS/install/setup.bash 不存在"

if [[ $HEADLESS -eq 0 && -z "${DISPLAY:-}" ]]; then
  die "DISPLAY 为空。请在 MobaXterm 的 SSH 终端里直接运行（不要套 screen/nohup），或加 --headless"
fi
if ! command -v xterm >/dev/null 2>&1 && [[ $HEADLESS -eq 0 ]]; then
  die "找不到 xterm，无法开窗口。请安装 xterm，或加 --headless"
fi

mkdir -p "$STATE" "$LOGROOT"
# 清掉上一轮的状态文件：pid 残留会让守护误判"进程已死"（踩过）
rm -f "$STATE"/*.pid "$STATE"/*.exit "$STATE"/*.suppress 2>/dev/null
rm -f "$STATE/cmd.restart" "$STATE/cmd.stop" "$STATE/started" 2>/dev/null

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOGROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_ID" "$LOGROOT/latest"

# 只保留最近 KEEP_RUNS 次
mapfile -t olds < <(find "$LOGROOT" -maxdepth 1 -type d -name 'run_*' -printf '%f\n' | sort | head -n -$KEEP_RUNS)
for o in "${olds[@]:-}"; do [[ -n "$o" ]] && rm -rf "$LOGROOT/$o"; done

cp "$PARAMS_FILE" "$RUN_DIR/params_snapshot.yaml"
cat > "$RUN_DIR/meta.json" <<EOF
{"run_id":"$RUN_ID","started_iso":"$(date -Iseconds)","params":"$PARAMS","params_file":"$PARAMS_FILE","workspace":"$WS","headless":$HEADLESS}
EOF

# ---------------------------------------------------------------- ROS 环境
# ROS 的 setup.bash 里有未定义变量，必须先关掉 -u 再 source
set +u
# shellcheck disable=SC1091
source /opt/tros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

export RDK_RUN_DIR="$RUN_DIR"
export RDK_STATE_DIR="$STATE"

log "工作区   : $WS"
log "参数文件 : $PARAMS_FILE"
log "日志目录 : $RUN_DIR"
log "面板端口 : ${PORT_HINT:-auto}"

# ---------------------------------------------------------------- 进程表
NAMES=(base camera yolo follow start panel safety collector)
TITLES=("①底盘" "②相机" "③YOLO" "④巡线+仲裁" "⑤发车" "⑥面板" "⑦兜底" "⑧DEC汇总")
WINDOWED=(1 1 1 1 1 0 0 0)          # 1=开 xterm 窗口
RESTARTABLE=(1 1 1 1 0 1 1 1)       # 1=允许自动重启
XTERM_PID=(); IS_ALIVE=(); RESTARTS=(); EXITCODE=(); HANDLED=(); LAUNCHED=()
for i in "${!NAMES[@]}"; do
  XTERM_PID[i]=""; IS_ALIVE[i]=0; RESTARTS[i]=0; EXITCODE[i]=""; HANDLED[i]=0; LAUNCHED[i]=0
done

cmd_string() {
  local n="$1"
  case "$n" in
    base)   echo "ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py" ;;
    camera) echo "cd '$WS/src/camera' && python3 -u camera.py" ;;
    yolo)   echo "cd '$WS/src/yolov8' && python3 -u bpu_yolov8_node.py" ;;
    follow) echo "ros2 launch line_follower line_follower_with_detection.launch.py params_file:='$PARAMS_FILE'" ;;
    start)  echo "ros2 run line_follower start_toggle_node" ;;
    panel)  echo "python3 -u '$TOOLS/panel/server.py'${PORT_HINT:+ --port $PORT_HINT}" ;;
    safety) echo "python3 -u '$TOOLS/safety_node.py'" ;;
    collector) echo "python3 -u '$TOOLS/dec_collector.py' --run-dir '$RUN_DIR'" ;;
  esac
}

idx_of() { local n="$1" k; for k in "${!NAMES[@]}"; do [[ "${NAMES[$k]}" == "$n" ]] && { echo "$k"; return; }; done; echo -1; }

launch_proc() {
  local n="$1"
  local i; i="$(idx_of "$n")"
  local cmd; cmd="$(cmd_string "$n")"
  LAUNCHED[i]=1
  rm -f "$STATE/$n.exit"

  if [[ "${WINDOWED[$i]}" == "1" && $HEADLESS -eq 0 ]]; then
    RDK_NAME="$n" RDK_CMD="$cmd" RDK_RUN_DIR="$RUN_DIR" RDK_STATE_DIR="$STATE" \
      xterm -T "${TITLES[$i]}" -geometry 104x26 -e "$TOOLS/wrap.sh" &
    XTERM_PID[i]=$!
    log "弹出窗口 ${TITLES[$i]}  ($n)"
  else
    RDK_NAME="$n" RDK_CMD="$cmd" RDK_RUN_DIR="$RUN_DIR" RDK_STATE_DIR="$STATE" \
      setsid nohup "$TOOLS/wrap.sh" >/dev/null 2>&1 &
    log "后台启动 $n"
  fi
}

# ---------------------------------------------------------------- 状态文件
write_status() {
  local tmp="$STATE/status.json.tmp"
  {
    printf '{"ts":%s,"run_id":"%s","params_file":"%s","global":"%s","started":%s,"procs":{' \
      "$(date +%s)" "$RUN_ID" "$PARAMS" "$GLOBAL" "$(if [[ $STARTED -eq 1 ]]; then echo true; else echo false; fi)"
    local first=1 i n
    for i in "${!NAMES[@]}"; do
      n="${NAMES[$i]}"
      [[ $first -eq 0 ]] && printf ','
      first=0
      local pid; pid="$(read_file "$STATE/$n.pid")"
      local ec="${EXITCODE[$i]}"
      printf '"%s":{"alive":%s,"restarts":%d,"pid":%s,"exit_code":%s}' \
        "$n" "$(if [[ ${IS_ALIVE[$i]} -eq 1 ]]; then echo true; else echo false; fi)" \
        "${RESTARTS[$i]}" "${pid:-0}" "${ec:-null}"
    done
    printf '}'
    if [[ -f "$STATE/panel_port" ]]; then
      printf ',"panel_port":%s' "$(read_file "$STATE/panel_port")"
    fi
    printf '}\n'
  } > "$tmp" && mv "$tmp" "$STATE/status.json"
}

# ---------------------------------------------------------------- 兜底停车
publish_zero() {
  local times="${1:-30}"
  timeout 5 ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" --times "$times" >/dev/null 2>&1 || true
}

GLOBAL="starting"
STARTED=0
STOPPING=0

shutdown_all() {
  [[ $STOPPING -eq 1 ]] && return
  STOPPING=1
  GLOBAL="stopping"
  write_status
  log "正在停止全部进程…"

  # 1) 趁底盘还活着先发零速
  publish_zero 30

  # 2) 按进程组杀（后台进程是 setsid 出来的，pid 就是组长，-pid 连带子进程）
  local i n pid
  for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"
    pid="$(read_file "$STATE/$n.pid")"
    if proc_alive "$pid"; then
      kill -TERM "-$pid" 2>/dev/null
      kill -TERM "$pid" 2>/dev/null
    fi
    if [[ -n "${XTERM_PID[$i]}" ]] && proc_alive "${XTERM_PID[$i]}"; then
      kill -TERM "${XTERM_PID[$i]}" 2>/dev/null
    fi
  done

  # 3) 兜底清扫：上一轮 start.sh 被 kill -9 时留下的孤儿进程
  sleep 0.5
  local pat
  for pat in "tools/wrap.sh" \
             "ros2 launch line_follower" \
             "ros2 launch turn_on_wheeltec_robot" \
             "install/line_follower/lib" \
             "camera.py" \
             "bpu_yolov8_node.py" \
             "tools/panel/server.py" \
             "tools/dec_collector.py" \
             "tools/safety_node.py"; do
    pkill -f "$pat" 2>/dev/null || true
  done

  GLOBAL="stopped"
  write_status
  log "已全部停止。日志在 $RUN_DIR"
}
trap 'shutdown_all; exit 0' INT TERM HUP

# ---------------------------------------------------------------- 启动
# 用“日志里出现就绪标记”来判启动完成：比 ros2 topic list 快一个数量级，
# 也不用起临时节点（每次调用约 1s，几十次会把启动拖到半分钟）。
wait_for_log() {
  local n="$1" pat="$2" tmo="${3:-15}" i=0 max=$(( ${3:-15} * 5 ))
  while (( i < max )); do
    if grep -qF -- "$pat" "$RUN_DIR/$n.log" 2>/dev/null; then
      log "$n 就绪：$pat"
      return 0
    fi
    sleep 0.2
    i=$(( i + 1 ))
  done
  log "$n 在 ${tmo}s 内未见 “$pat”，继续启动后续（请查 $RUN_DIR/$n.log）"
  return 1
}

launch_proc base
wait_for_log base "serial port opened" 20

launch_proc camera
wait_for_log camera "摄像头已成功启动" 12

launch_proc yolo
wait_for_log yolo "视觉检测节点就绪" 20

launch_proc follow
wait_for_log follow "Line follower subscribing" 20

launch_proc start

FOLLOW_GRACE_UNTIL=$(( $(date +%s) + FOLLOW_GRACE ))

# 面板是后台进程，Ctrl+C 时可能来不及退出；残留实例会一直占着 8080-8089，
# 攒到 10 个就再也起不来（表现为 exit code 134 + "8080-8089 全部被占用"）。
if [[ $NO_PANEL -eq 0 ]]; then
  if pgrep -f "tools/panel/server.py" > /dev/null 2>&1; then
    log "清理残留 panel 进程"
    pkill -f "tools/panel/server.py" 2>/dev/null || true
    sleep 1
  fi
  launch_proc panel
fi
launch_proc safety
launch_proc collector

GLOBAL="running"
write_status
log "启动完成。发车请在 ⑤发车 窗口按【空格】；停止请在本窗口 Ctrl+C。"

# ---------------------------------------------------------------- 守护主循环
LAST_NODE_CHECK=0
while [[ $STOPPING -eq 0 ]]; do
  sleep 1

  # ---- 发车状态由面板写盘（不要在这里起 ros2 进程，每秒一次太贵）
  if [[ -f "$STATE/started" ]]; then
    [[ "$(read_file "$STATE/started")" == "1" ]] && STARTED=1 || STARTED=0
  fi

  # ---- 面板请求的重启
  if [[ -f "$STATE/cmd.restart" ]]; then
    want="$(read_file "$STATE/cmd.restart")"
    rm -f "$STATE/cmd.restart"
    if [[ -n "$want" && "$(idx_of "$want")" != "-1" ]]; then
      i="$(idx_of "$want")"
      log "面板请求重启 $want"
      printf '[DEC] panel restart name=%s\n' "$want"
      pid="$(read_file "$STATE/$want.pid")"
      proc_alive "$pid" && kill -TERM "$pid" 2>/dev/null
      sleep 1
      RESTARTS[i]=0
      launch_proc "$want"
      [[ "$want" == "follow" ]] && FOLLOW_GRACE_UNTIL=$(( $(date +%s) + FOLLOW_GRACE ))
    fi
  fi

  # ---- 进程存活与重启
  for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"
    [[ ${LAUNCHED[$i]} -eq 1 ]] || continue      # 没启动过的（如 --no-panel）不监管
    pid="$(read_file "$STATE/$n.pid")"
    if proc_alive "$pid"; then
      IS_ALIVE[i]=1
      HANDLED[i]=0
      continue
    fi

    # 已经处理过这次死亡就跳过（重启后 pid 会变，HANDLED 会被复位）
    if [[ ${HANDLED[$i]} -eq 1 ]]; then continue; fi
    HANDLED[i]=1
    IS_ALIVE[i]=0
    ec="$(read_file "$STATE/$n.exit")"
    EXITCODE[i]="${ec:-129}"

    if [[ -f "$STATE/$n.suppress" ]]; then
      rm -f "$STATE/$n.suppress"
      log "$n 被手动停止，不重启"
      continue
    fi

    case "${ec:-129}" in
      0)
        printf '[DEC] proc_exit name=%s code=0\n' "$n"; log "$n 正常退出，不重启" ;;
      129)
        printf '[DEC] proc_exit name=%s code=129\n' "$n"; log "$n 窗口被关闭，不重启" ;;
      *)
        printf '[DEC] proc_down name=%s code=%s\n' "$n" "$ec"
        log "\033[33m$n 异常退出 code=$ec\033[0m"
        if [[ "${RESTARTABLE[$i]}" != "1" ]]; then
          log "$n 不在重启名单内"
          continue
        fi
        RESTARTS[i]=$(( ${RESTARTS[i]} + 1 ))
        if [[ ${RESTARTS[i]} -gt $MAX_RESTARTS ]]; then
          log "\033[31m$n 重启次数超过 $MAX_RESTARTS，停止全部\033[0m"
          printf '[DEC] restart_exhausted name=%s\n' "$n"
          GLOBAL="restart_exhausted"
          write_status
          shutdown_all
          exit 1
        fi
        publish_zero 20
        log "重启 $n（第 ${RESTARTS[i]} 次）"
        launch_proc "$n"
        [[ "$n" == "follow" ]] && FOLLOW_GRACE_UNTIL=$(( $(date +%s) + FOLLOW_GRACE ))
        ;;
    esac
  done

  # ---- ④ 的子节点检查（R02）
  now=$(date +%s)
  if [[ $now -ge $FOLLOW_GRACE_UNTIL ]] && (( now - LAST_NODE_CHECK >= NODE_CHECK_EVERY )); then
    LAST_NODE_CHECK=$now
    fi_idx="$(idx_of follow)"
    if [[ ${IS_ALIVE[$fi_idx]} -eq 1 ]]; then
      # 不用 ros2 node list：CLI 启动要 1~4s，系统忙时会误判成“节点缺失”。
      # 直接数进程，瞬间完成且不依赖 ROS daemon。
      cnt=0
      for exe in line_follower_node detection_turn_controller cmd_mux_node; do
        if pgrep -f "install/line_follower/lib/line_follower/$exe" >/dev/null 2>&1; then
          cnt=$(( cnt + 1 ))
        fi
      done
      if [[ "${cnt:-0}" -lt 3 ]]; then
        log "\033[33m④ 巡线节点缺失（$cnt/3），重启整个 launch\033[0m"
        printf '[DEC] follow_nodes_missing have=%s\n' "${cnt:-0}"
        pid="$(read_file "$STATE/follow.pid")"
        proc_alive "$pid" && kill -TERM "$pid" 2>/dev/null
        sleep 1
        RESTARTS[fi_idx]=$(( ${RESTARTS[fi_idx]} + 1 ))
        if [[ ${RESTARTS[fi_idx]} -gt $MAX_RESTARTS ]]; then
          log "\033[31m④ 重启次数超过 $MAX_RESTARTS，停止全部\033[0m"
          EXITCODE[fi_idx]=99
          shutdown_all; exit 1
        fi
        publish_zero 20
        launch_proc follow
        FOLLOW_GRACE_UNTIL=$(( $(date +%s) + FOLLOW_GRACE ))
      fi
    fi
  fi

  write_status
done
