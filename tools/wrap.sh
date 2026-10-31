#!/usr/bin/env bash
# 单进程包装器：xterm / nohup 都通过它启动，
# 负责把 stdout+stderr 同时写到日志、把退出码写盘供 start.sh 判定。
#
# 通过环境变量传参（避免多层引号）：
#   RDK_NAME     进程名
#   RDK_CMD      要执行的命令
#   RDK_RUN_DIR  日志目录
#   RDK_STATE_DIR 状态目录
#
# 日志里的两条分界线（重启后靠它们区分新旧进程，别删）：
#   [wrap] ===== <name> start pid=... =====   人看的分界
#   [DEC] proc start name=<name> pid=...      事件流/面板用的结构化分界

set -o pipefail

name="${RDK_NAME:?RDK_NAME required}"
cmd="${RDK_CMD:?RDK_CMD required}"
run_dir="${RDK_RUN_DIR:?RDK_RUN_DIR required}"
state_dir="${RDK_STATE_DIR:?RDK_STATE_DIR required}"

log="$run_dir/$name.log"
exit_file="$state_dir/$name.exit"
seg_file="$state_dir/$name.seg"

mkdir -p "$run_dir" "$state_dir"
echo $$ > "$state_dir/$name.pid"
rm -f "$exit_file"

# 记下本次进程在日志里的起点。面板据此只读“这一次启动”的那一段，
# 也据此判断 YAML 是否比进程新（= 改了参数但没重启）。
log_off=0
[[ -f "$log" ]] && log_off="$(stat -c%s "$log" 2>/dev/null || echo 0)"
{
  printf 'pid=%s\n' "$$"
  printf 'log_off=%s\n' "$log_off"
  printf 'started_epoch=%s\n' "$(date +%s)"
  printf 'started_iso=%s\n' "$(date -Iseconds)"
} > "$seg_file"

# 分界线必须写进日志文件本身；只打印到窗口的话，重启后两段输出会混在一起。
{
  printf '\n[wrap] ===== %s start pid=%s at %s =====\n' "$name" "$$" "$(date '+%F %T')"
  printf '[DEC] proc start name=%s pid=%s\n' "$name" "$$"
} >> "$log"

# 退出时落盘退出码。关窗口时 bash 收到 SIGHUP，退出码 129。
trap 'rc=$?; echo "$rc" > "$exit_file"; printf "[DEC] proc exit name=%s rc=%s\n" "$name" "$rc" >> "$log"; echo "[wrap] $name exited rc=$rc"' EXIT

echo "[wrap] $name start pid=$$ at $(date '+%F %T')"

# tee 默认会攒一大块才写文件，面板看到的就是“慢半拍”的日志。
# stdbuf -oL 强制行缓冲，日志落盘和窗口输出同步（R18）。
TEE="tee"
command -v stdbuf >/dev/null 2>&1 && TEE="stdbuf -oL tee"
eval "$cmd" 2>&1 | $TEE -a "$log"
