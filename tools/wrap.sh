#!/usr/bin/env bash
# 单进程包装器：xterm / nohup 都通过它启动，
# 负责把 stdout+stderr 同时写到日志、把退出码写盘供 start.sh 判定。
#
# 通过环境变量传参（避免多层引号）：
#   RDK_NAME     进程名
#   RDK_CMD      要执行的命令
#   RDK_RUN_DIR  日志目录
#   RDK_STATE_DIR 状态目录

set -o pipefail

name="${RDK_NAME:?RDK_NAME required}"
cmd="${RDK_CMD:?RDK_CMD required}"
run_dir="${RDK_RUN_DIR:?RDK_RUN_DIR required}"
state_dir="${RDK_STATE_DIR:?RDK_STATE_DIR required}"

log="$run_dir/$name.log"
exit_file="$state_dir/$name.exit"

mkdir -p "$run_dir" "$state_dir"
echo $$ > "$state_dir/$name.pid"
rm -f "$exit_file"

# 退出时落盘退出码。关窗口时 bash 收到 SIGHUP，退出码 129。
trap 'rc=$?; echo "$rc" > "$exit_file"; echo "[wrap] $name exited rc=$rc"' EXIT

echo "[wrap] $name start pid=$$ at $(date '+%F %T')"
eval "$cmd" 2>&1 | tee -a "$log"
