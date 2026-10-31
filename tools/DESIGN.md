# RDK X5 智能车 · 运行工具链设计（start.sh + 安全兜底 + Web 面板）

> 版本：v1.0　编写日期：2026-10-20　状态：**待评审，未动工**
>
> 本文档描述的是**工具链**（怎么跑、怎么看、怎么调），不涉及巡线/决策算法本身的改造。
> 算法侧改动另见后续文档。

---

## 1. 目标与非目标

### 1.1 目标

| # | 目标 | 衡量标准 |
|---|---|---|
| G1 | 一条命令拉起全部 5 个进程（并行，独立可见） | `./tools/start.sh` 后弹出 5 个 xterm 窗口 |
| G2 | 任一进程崩溃能自动恢复，且**车不会失控** | kill 掉巡线节点，车在 0.3 s 内停下，节点被重启 |
| G3 | 浏览器里能看调试图、看决策日志、大赛前改参数 | `http://localhost:8080` |
| G4 | 赛中面板强制只读（合规） | 发车后所有写接口返回 423 |
| G5 | 每次运行留下可对比的日志 | `tools/logs/run_<时间戳>/` |

### 1.2 非目标（明确不做）

- ❌ 不做赛道定位 / SLAM / 实时轨迹可视化（没有定位数据，做不了）
- ❌ 不做参数热加载（现有代码 90% 参数是启动快照，见 R07）
- ❌ 不做面板发车按钮（会和 `start_toggle_node` 打架，见 R05）
- ❌ 不改巡线算法、不改 YOLO 推理逻辑
- ❌ 不引入 flask / node / rosbridge 等外部依赖

---

## 2. 已锁定的决策

| 编号 | 决策 | 结论 |
|---|---|---|
| D1 | 面板在赛中的写入权限 | **赛前可写 / 赛中只读**（规则禁止手动遥控） |
| D2 | "5 个并行"的形态 | **5 个独立 xterm 窗口**，各自可交互 |
| D3 | 面板技术栈 | **Python 标准库 `http.server` 自研**，零第三方依赖 |
| D4 | 面板内容 | 调试图 MJPEG + 参数表单 + 日志流 + 重启按钮 |
| D5 | 改参数如何生效 | **写 YAML + 手动重启该进程**（不热加载） |
| D6 | 自动重启语义 | 异常退出 → 先发零速 → 重启，**最多 3 次**，用完全停 |
| D7 | 决策日志 | 先做 `[DEC]` 前缀（B），后续再做结构化话题（C） |
| D8 | 网络 | Clash 端口 **7897**；板子↔Windows **不在同一二层网**，走 SSH 隧道 |
| D9 | 面板端口 | **8080**，被占则 8081/8082 递推 |
| D10 | 面板监听地址 | **仅 127.0.0.1**（靠 SSH 隧道访问，零暴露面） |
| D11 | 日志保留 | 最近 10 次运行，更早自动删除 |
| D12 | 自动重启判据 | 仅 **退出码非 0 且非 129** 才重启（0=正常退出，129=关窗口） |
| D13 | 决策日志落盘 | `[DEC]` 同时汇总成 **`events.csv`** |
| D14 | 参数文件命名 | `fork.yaml` / `roundabout.yaml` |

---

## 3. 现状事实（已在本机核实，2026-10-20）

### 3.1 环境

```
工作区          /home/sunrise/ws_rdkx5_1
ROS             /opt/tros/humble（必须用它，用 /opt/ros/humble 会缺 ai_msgs）
板子网卡        wlan0 10.151.31.25/24（唯一非 lo 接口），网关 10.151.31.83
CPU / 内存      8 核 / 7 GB（已用约 1.5 GB）
外网直连        pypi.org 6 秒超时 → 不通
依赖自检        yaml / cv2 / rclpy / numpy / http.server / threading 全部 OK
端口            8080-8089 空闲
```

### 3.2 现有进程与话题

| 进程 | 订阅 | 发布 |
|---|---|---|
| `turn_on_wheeltec_robot` | `/cmd_vel` | `/odom` `/odom_combined` `/imu/data` `/PowerVoltage` |
| `camera.py` | — | `/image_raw`（640×480 bgr8 @30） |
| `bpu_yolov8_node.py` | `/image_raw` | `/hobot_dnn_detection` `/yolo_detection_image` |
| `line_follower_node` | `/image_raw` `/line_follower/start_cmd` | `/cmd_vel_follower` `/cmd_vel_stop` `line_follower/debug_image` `line_follower/track_state` `line_follower/centroid` |
| `detection_turn_controller` | `/hobot_dnn_detection` `line_follower/centroid` `line_follower/track_state` | `/cmd_vel_detection` |
| `cmd_mux_node` | 上面三个 cmd_vel | `/cmd_vel` |
| `start_toggle_node` | 键盘 stdin | `line_follower/start_cmd`（**10 Hz 持续发布**） |

### 3.3 相关源码位置

```
src/line_follow_ws/src/line_follower/src/line_follower_node.cpp      # 巡线主体，2010 行
src/line_follow_ws/src/line_follower/src/detection_turn_controller.cpp
src/line_follow_ws/src/line_follower/src/cmd_mux_node.cpp
src/line_follow_ws/src/line_follower/src/start_toggle_node.cpp
src/line_follow_ws/src/line_follower/config/line_follower_params.yaml
src/line_follow_ws/src/line_follower/launch/line_follower_with_detection.launch.py
src/camera/camera.py
src/yolov8/bpu_yolov8_node.py
```

---

## 4. 网络与访问

### 4.1 实测结论

```
板子 → 192.168.5.82 的路由:  via 10.151.31.83 dev wlan0    ← 要过网关，非直连
不走代理 → pypi.org:                        超时
走代理   → http://192.168.5.82:7897:        超时
```

结论：**板子无法直接访问 Windows 的 Clash**。必须用 SSH 隧道。

### 4.2 两条隧道（MobaXterm 会话设置 → SSH → Tunneling）

| 类型 | 远程端口 | 本地主机 | 本地端口 | 作用 |
|---|---|---|---|---|
| **Remote** forwarding | `7897` | `localhost` | `7897` | 板子 `127.0.0.1:7897` → Windows 的 Clash |
| **Local** forwarding | `8080`（板子侧） | `localhost` | `8080` | Windows 浏览器 `localhost:8080` → 板子面板 |

方向相反，配反了就打不开。

> ⚠️ **校验方法**：连上后板子执行 `curl -I -m 5 https://pypi.org`，返回 `HTTP/2 200` 才算隧道通。
> 不通时先确认 Remote forwarding 的"远程端口"填的是 **7897**（不是 7890）。

### 4.3 板子侧环境变量

写进 `start.sh` 头部，同时建议加进 `~/.bashrc`：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=$http_proxy
export no_proxy=127.0.0.1,localhost,10.151.31.0/24
```

> ⚠️ `no_proxy` 不能省，否则面板自己访问 `127.0.0.1:8080`、ROS 的本地回环都可能被塞进代理（见 R12）。

---

## 5. 目录结构

```
/home/sunrise/ws_rdkx5_1/tools/
├── start.sh                 # 唯一入口
├── DESIGN.md                # 本文档
├── panel/
│   ├── server.py            # 面板后端，零依赖
│   ├── index.html           # 单文件前端，无外链
│   └── params_schema.py     # 参数分组/单位/范围/说明
├── safety_node.py           # deadman 兜底节点
└── logs/
    ├── run_20261020_2130/
    │   ├── base.log  camera.log  yolo.log  follow.log  start.log  panel.log
    │   └── meta.json        # 启动时间、参数文件快照
    └── latest -> run_20261020_2130

/tmp/rdkx5/                  # 运行时状态（易失，重启即清）
├── status.json              # start.sh 每秒写的进程状态，面板读它
├── cmd.restart              # 面板 → start.sh 的重启请求通道
└── cmd.stop                 # 面板 → start.sh 的停止请求通道
```

> 放 `tools/` 而不是 `src/`：`colcon` 会扫 `src/`，别让构建系统看见这些文件。

---

## 6. P0 — `start.sh`：一条命令，5 个并行窗口

### 6.1 用法

```bash
./tools/start.sh                  # 全部拉起（5 窗口 + 面板 + 兜底节点）
./tools/start.sh --no-panel       # 只起 5 个控制进程，调试用
./tools/start.sh --params fork    # 指定参数文件前缀（岔路图）
./tools/start.sh --params roundabout
```

> ⚠️ **必须在 MobaXterm 的 SSH 终端里直接执行**，不能套 `screen`。
> `screen` 会话里 `DISPLAY` 通常为空，`xterm` 弹不出来（见 R03）。

### 6.2 进程表

| # | 角色 | 启动命令 | 日志 | 自动重启 |
|---|---|---|---|---|
| ① | 底盘 | `ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py` | `base.log` | ✅ |
| ② | 相机 | `python3 src/camera/camera.py` | `camera.log` | ✅ |
| ③ | YOLO | `python3 src/yolov8/bpu_yolov8_node.py` | `yolo.log` | ✅ |
| ④ | 巡线+转向+仲裁 | `ros2 launch line_follower line_follower_with_detection.launch.py params_file:=<选定>` | `follow.log` | ✅（按节点判定） |
| ⑤ | 发车开关 | `ros2 run line_follower start_toggle_node` | `start.log` | ❌（重开会丢发车状态） |
| ⑥ | 面板 | `python3 tools/panel/server.py` | `panel.log` | ✅（后台，无窗口） |
| ⑦ | 兜底 | `python3 tools/safety_node.py` | `safety.log` | ✅（后台，无窗口） |
| ⑧ | DEC 汇总 | `python3 tools/dec_collector.py` | `collector.log` + `events.csv` | ✅（后台，无窗口） |

每个窗口命令形如：

```bash
xterm -T "①底盘" -geometry 100x28 -e bash -lc \
  "source /opt/tros/humble/setup.bash; source install/setup.bash; \
   ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py 2>&1 | tee -a $LOG"
```

### 6.3 启动顺序（有依赖，但进程是并发的）

```
①底盘 ──(等 /odom 出现, 最长 15s)──> ②相机 ──(等 /image_raw, 最长 10s)──> ③YOLO ──> ④巡线 ──> ⑤发车
                                                                     ↓
                                                          ⑥面板 + ⑦兜底（与上面全并行）
```

等待用轮询实现，不写死 `sleep`：

```
轮询 `ros2 topic hz /odom`（带 1s 超时）出现输出 → 认为底盘就绪 → 继续
超时 → 日志打 [WARN] base not ready after 15s，仍然继续启动后续（人要能看到失败）
```

### 6.4 守护与重启规则

```
主循环（每 1s）:
  for each 受管进程:
      if 进程存活: 继续
      else:
          读 <name>.exit（不存在则视为 129，即被 SIGHUP 杀掉）
          if 全局停止中: 不重启
          elif code == 0:    [DEC] proc_exit name=<x> code=0    # 正常退出，不重启
          elif code == 129:  [DEC] proc_exit name=<x> code=129  # 关窗口，视为主动停止
          else:
              [DEC] proc_down name=<x> exit=<code>
              发零速 20 帧（1s，保险起见，主要靠 safety_node）
              重启次数 +1
              if 重启次数 > 3:
                  停止全部 + 面板红字 + 退出码 1
              else: 重新拉起并弹窗
```

**退出码判据（D12 已确认）**：

| 退出码 | 含义 | 是否重启 |
|---|---|---|
| `0` | 进程自己正常退出 | ❌ |
| `129`（128+SIGHUP） | xterm 窗口被关闭 | ❌（视为主动停止） |
| 其它非零 | 崩溃/被杀/异常 | ✅（最多 3 次） |

> 注意 `130`（SIGINT）**会**触发重启——那是正常的 Ctrl+C 停止方式，应该走 `start.sh` 的全局停止，而不是在单个窗口里按。文档与窗口标题里都会提示这一点。

**④ 的特殊判定**：`ros2 launch` 是父进程，子节点崩掉父进程不会退出。所以对 ④ 的判断不是"进程在不在"，而是：

```bash
ros2 node list | grep -c "line_follower_node\|detection_turn_controller\|cmd_mux_node"
# 必须等于 3，少于 3 判定为异常 → 杀掉整个 launch → 重启
```

**全局停止**：`start.sh` 捕获 Ctrl+C / SIGTERM → ① 发 30 帧零速 → ② kill 全部 → ③ 写 `status.json` 标记 stopped。

### 6.5 状态文件（给面板读）

`start.sh` 每秒写 `/tmp/rdkx5/status.json`：

```json
{
  "ts": 1792500388.29,
  "run_id": "run_20261020_2130",
  "params_file": "fork",
  "global": "running",
  "procs": {
    "base":    {"alive": true, "restarts": 0, "pid": 1234},
    "camera":  {"alive": true, "restarts": 0, "pid": 1235},
    "yolo":    {"alive": true, "restarts": 0, "pid": 1236},
    "follow":  {"alive": true, "restarts": 0, "nodes": ["line_follower_node","detection_turn_controller","cmd_mux_node"]},
    "start":   {"alive": true, "restarts": 0, "pid": 1238},
    "panel":   {"alive": true, "restarts": 0, "pid": 1240},
    "safety":  {"alive": true, "restarts": 0, "pid": 1241}
  },
  "started": false,
  "estop": false
}
```

`started` 由面板订阅 `line_follower/start_cmd` 得到（不是 `start.sh` 判断的）。

### 6.6 验收

- [ ] 5 个 xterm 窗口全部弹出，标题正确
- [ ] 每个窗口有输出，`tools/logs/latest/*.log` 有内容
- [ ] `ros2 topic hz /odom` 有 20+ Hz，`/image_raw` 30 Hz
- [ ] 手动 `kill` 掉 `line_follower_node` → 5 秒内 ④ 窗口重新出现节点日志，`status.json` 的 `restarts` 变成 1
- [ ] 手动 `kill` 掉 ④ 四次 → 全部停止 + 红字提示

---

## 7. P1 — `safety_node.py`：失控兜底（deadman）

### 7.1 为什么必须有

**下位机固件没有超时保护**（`src/rdkx5.md` 第 9 节已记录）：主控不再发 `/cmd_vel` 时，车会**保持最后一条速度指令一直跑**。

`line_follower` 崩溃 → `/cmd_vel_follower` 静默 → `cmd_mux` 不再输出 → 车以 0.3 m/s 直冲，直到撞墙或冲出赛道。

用 bash 补救来不及：`ros2 topic pub --times 20` 冷启动要 1 s 以上，0.3 m/s 已经冲出 30 cm。

### 7.2 设计

```
        订阅 /cmd_vel_follower    ─┐
        订阅 /cmd_vel_detection   ─┼─> 三者全部静默 > 300ms ?
        订阅 /cmd_vel_stop        ─┘         │
                                             ├─ 是 → 以 20 Hz 向 /cmd_vel 发零速
                                             └─ 否 → 停止兜底，什么都不发
```

**关键点：订阅的是三个上游话题，不是 `/cmd_vel` 本身**。否则自己发的零速会被自己收到，逻辑自锁。

### 7.3 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `silence_timeout` | 0.30 s | 三路全静默判定阈值 |
| `publish_rate` | 20 Hz | 兜底发布频率 |
| `topics` | 见上 | 可配置 |

### 7.4 日志

```
[DEC] safety deadman_on  silence=0.42s
[DEC] safety deadman_off last_src=cmd_vel_follower
```

### 7.5 验收

- [ ] 正常跑车时 `safety_node` **完全静默**（`ros2 topic echo /cmd_vel` 看不到重复零速）
- [ ] 按空格发车后 `kill` 掉 `line_follower_node` → 车在约 0.3 m/s × 0.3 s ≈ 10 cm 内停下
- [ ] 重启完成后车自动恢复行驶（因为 mux 恢复输出，兜底自动解除）

> ⚠️ 这里有个**预期行为**：节点被重启后，`line_follower_node` 的 `start_enabled_` 是 false（重启即复位），所以车会停住等人再按一次空格。这是**设计如此**，不是 bug（见 R04）。

---

## 8. P2 — 面板只读版

### 8.1 进程模型

```
server.py
├── 主线程：ThreadingHTTPServer 监听 127.0.0.1:8080
├── ROS 线程：rclpy.spin(node)
│     ├─ 订阅 line_follower/debug_image   → 最新的 JPEG 字节（质量 70）
│     ├─ 订阅 /yolo_detection_image       → 最新的 JPEG 字节
│     ├─ 订阅 line_follower/start_cmd     → started 状态
│     └─ 订阅 line_follower/track_state   → 当前状态字符串
└── 文件线程：每秒读 /tmp/rdkx5/status.json（不存在则返回 global=down）
```

> 面板**不参与任何控制链路**。面板崩溃只影响看，不影响车（R10）。

### 8.2 HTTP 路由表

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | `index.html` |
| GET | `/api/status` | 合并 `status.json` + ROS 侧状态 |
| GET | `/api/stream/debug` | MJPEG，**限 6 fps** |
| GET | `/api/stream/yolo` | MJPEG，**限 6 fps** |
| GET | `/api/logs?f=follow&dec=1&offset=N` | 增量日志，返回 `{lines:[], offset:N}` |
| GET | `/api/params?map=fork` | 解析 YAML 成表单结构 |
| POST | `/api/params` | 写 YAML（**仅未发车时**，否则 423） |
| POST | `/api/restart/<proc>` | 写 `cmd.restart`（**仅未发车时**） |
| POST | `/api/estop` | **任何状态可用**，发 30 帧零速到 `/cmd_vel_stop` |
| GET | `/api/timer` | `{started_at, elapsed, stopped}` |

### 8.3 MJPEG 实现要点

```
HTTP 响应头: Content-Type: multipart/x-mixed-replace; boundary=frame
循环: 取最新 JPEG → 写 boundary + Content-Length + 数据 → sleep 到 6fps 节奏
```

- 帧率**硬限 6 fps**：板子要留 CPU 给 30 fps 的巡线循环，面板抢 CPU 的直接后果是车抖（R09）
- JPEG 质量 **70**，640×480 约 25 KB/帧，6 fps ≈ 150 KB/s，隧道压力可忽略
- **页面关闭时生成器必须退出**：靠写 socket 抛异常来终止，否则线程泄漏

### 8.4 页面布局

```
┌────────────────────────────────────────────────────────────┐
│ ●底盘 ●相机 ●YOLO ●巡线 ●发车   [黄线 2/4]                │
│ 计时 00:37.4   估算分 +15   状态 tracking  [🔴 立即停车]   │
├──────────────────────┬─────────────────────────────────────┤
│  巡线调试图          │ 参数：地图 [岔路 ▾]                 │
│  ┌────────────────┐  │ ├ 速度                              │
│  │  (MJPEG)       │  │ │  target_speed  0.30 [____]        │
│  └────────────────┘  │ │  kp            2.45 [____]        │
│  ┌────────────────┐  │ ├ 视觉                              │
│  │  YOLO 检测图   │  │ │  roi_top_ratio 0.65 [____]        │
│  └────────────────┘  │ ├ 黄线 / 红锥 / 丢线 ...            │
│                      │ [保存] [重启巡线]  ← 发车后置灰     │
├──────────────────────┴─────────────────────────────────────┤
│ 日志  [仅决策 ☑] [全量 ☐]  搜索 [________]  自动滚动 ☑     │
│ [DEC] yellow count=2/4 stop_begin t=18.32                  │
│ [DEC] cone side=L risk=0.42 yaw_bias=-0.31                 │
└────────────────────────────────────────────────────────────┘
```

前端单文件，**不引任何 CDN**（外网不稳，且要让录像里不出现加载失败）。

### 8.5 验收

- [ ] 浏览器 `localhost:8080` 打开，五灯全绿
- [ ] 两路图像 6 fps 流畅（不是幻灯片）
- [ ] 日志区按 `[DEC]` 过滤生效
- [ ] 关掉页面后板子 `top` 里 server.py 的 CPU 回落（证明生成器退出了）

---

## 9. P3 — 参数编辑与受控重启

### 9.1 生效链路（对应 D5）

```
面板表单 ──POST /api/params──> 备份 .bak → 校验范围 → 写 YAML
                                              ↓
用户点 [重启巡线] ──写 /tmp/rdkx5/cmd.restart──> start.sh 守护读到
                                              ↓
                        kill ④ 的 launch → 重新 launch（带 params_file）
```

**为什么重启请求要绕一圈文件、而不是面板直接 kill**：面板不是这些进程的父进程，直接 kill 会造成僵尸/父子关系混乱，且 `start.sh` 的重启计数会记错。让唯一的管家（`start.sh`）动手。

### 9.2 参数分组（`params_schema.py`）

| 组 | 参数 | 单位 | 范围 |
|---|---|---|---|
| 分段（逐段） | `segN_` + 弯道 14 键（含 `steer_bias`，见 9.2.1） | — | — |
| 分段切换 | `seg2_start_count` `seg3_start_count` `seg4_start_dist_m` | 次, m | 1–4 / 1–6 |
| 速度限幅 | `max_speed` `min_speed`（全段共用） | m/s | 0–1.5 |
| 视觉 | `binary_threshold` `white_s_max` `white_v_min` `morph_kernel` `morph_iterations` | — | 0–1 / 0–255 |
| 黄线 | `yellow_stop_duration` `yellow_permanent_count` `yellow_area_ratio_threshold` `yellow_clear_frames` | s, 次 | — |
| 红锥 | `red_near_ratio` `red_very_close_ratio` `red_soft_yaw` `red_repulsion_gain` `red_min_forward_speed` `red_corridor_half_width_ratio` | — | 0–1 |
| 丢线 | `line_lost_stop_time` `line_lost_forward_speed` `line_lost_yaw_hold_factor` | s, m/s | — |

每项带上现在 YAML 里的值和代码里的注释，说明它管什么。

### 9.2.1 分段参数（每段一套）

**为什么分段**：同一套 `kp` / `roi` / 目标速度不可能同时适合起步直道、选支转弯和半径 0.48 m 的环岛。
分段之后每段独立的 14 个键，面板按段切换编辑。

| 代码 | 作用 |
|---|---|
| `include/line_follower/segment_planner.hpp` | `SegmentParams`（14 字段，含 `steer_bias`）、`SegmentPlan`（段数 + 边界）、`SegmentPlanner::update()`；零 ROS 依赖 |
| `line_follower_node.cpp` | `load_segment_parameters()` 读 `segN_*`；`apply_segment()` 把当前段**镜像**到原有的 `target_speed_ / kp_ / steer_bias_ / max_yaw_rate_ / roi_* / speed_gain_damping_ / steer_ff_*`，热循环与红锥避让一行不改 |
| `odom_callback` | `|v|·dt` 积分成 `dist_since_yellow_`（dt 限 0–0.2 s），每根黄线归零 |
| `image_callback` | 最前面调 `update_segment()`（ROI 逐段，必须在 `process_image` 之前）；落后一帧 |

**切换规则**（`track_mode` 决定 3 段还是 4 段）。边界有两种，填 0 = 该条件关闭：

| 地图 | 段 | 进入条件 |
|---|---|---|
| fork | 1 → 2 | 跨过第 `seg2_start_count`（默认 2）根黄线 |
| fork | 2 → 3 | 第 2 根黄线之后 `seg3_start_dist_m`（选支/转弯结束） |
| roundabout | 1 → 2 / 2 → 3 | 跨过第 2 / 3 根黄线 |
| roundabout | 3 → 4 | 第 3 根黄线之后 `seg4_start_dist_m`（默认 3.30 m） |

**选支（岔路）：`steer_bias`** —— 这是唯一能让车"故意偏到线的一侧"的参数。
`kp` / `roi` / `damping` 全是"跟着线走"的量，P 控制器稳态永远是线在画面中间。

```
yaw_cmd = clamp(-(kp·error + ff_gain·ff) + steer_bias, -max_yaw_rate, +max_yaw_rate)
                                        ↑ rad/s，负 = 向右
稳态：kp·error = steer_bias  ->  error = -bias/kp（线固定在画面偏一侧）
```

调法：段2 的 `steer_bias` 从 **-0.40** 起、一档 0.10 加；同时把该段 `target_speed` 降到
0.25–0.30、`kp` 降到 2.0–2.5（kp 高会把偏置顶回去），`steer_ff_gain` 置 0（别和前馈打架）。
`|bias|` 必须小于 `max_yaw_rate`，否则只剩饱和打角（启动会告警并截断）。
进分支稳了以后，用 `seg3_start_dist_m` 把参数切回 `steer_bias=0` 的段3。

**段内两段偏置（拐进去 → 间隔 → 拐出来）**：选支动作很短（0.3–0.6 m），一次偏置进得去但出不来，
所以在同一段里打两个短脉冲，`steer_bias_at()` 实现，参数在段2（`SEGMENT_EXTRA`）：

```
0 .. D1             B1 = steer_bias          （拐进支路）
D1 .. D1+G          0                        （间隔：靠 P 跟线通过）
D1+G .. D1+G+D2     B2 = steer_bias2         （B2 与 B1 同号=再拐一次，反号=拐回来）
之后                0
D1<=0：整段 B1；D2<=0：与 D1 等长（对称脉冲）
```

**量级换算**（调参必看）：偏置是持续施加的角速度，不是脉冲，

```
累计转角 Δψ ≈ B × (D / v)          D=里程窗口，v=车速
0.2 m @ 0.3 m/s @ B=-0.7  ->  ≈ 27°（再被舵机 τ≈0.25 s 的滞后削弱一部分）
```

所以：窗口 0.2–0.4 m、车速 0.25–0.35 m/s 最舒服；窗口短于 ~0.2 s（0.3 m/s 下 6 cm）
会被一阶滞后吃掉，等于没打。岔路段2 的初值：`-0.70 / 0.20 / 0.15 / -0.70 / 0.20`。

段号在一次运行内**单调递增**（终点黄线会把里程清零，不单调就会倒回段 3），只有换圈时 `reset()` 回退。

**段 4 的 3.30 m 怎么来的**：几何出口距第 3 根黄线 2.27 m（环岛二 κ>1.0 区间 6.78–8.68，半径 0.48–0.50 m，扫过约 221°），
黄线检测提前量 0.35–0.70 m，odom 刻度 ×1.08，再留 0.15 余量。**宁可切晚**：在弯里切会切内线，晚切只损失约 0.2 s。
首跑校准：看视频里车头压上第 3 根黄线那一帧的 `d`（调试图 `SEG x/4 d=…` 或 `[DEC] seg d=`），
`阈值 = 2.27×1.08 + d + 0.15`。

**否决过的方案**：

| 方案 | 为什么不用 |
|---|---|
| 画面白占比 | 实测反向：环岛内 70–90%，出口直道 68%→60%（直路收敛到灭点白像素更少） |
| 转角/`angular.z` 变小 | 它测的是**跟踪误差**不是**路弯度**；环岛内跟得准时转角也是 0，6.6–6.8 m 处实测为 0 |
| 视觉弯曲代理做扳机 | 有饱和死区（远带占满整幅时中点恒为 320，8.2–8.4 m 处假"直"），需要额外判别，收益不抵复杂度 |

### 9.3 两套参数文件

`fork.yaml`（岔路图）/ `roundabout.yaml`（环岛图），从现有 `line_follower_params.yaml` 复制拆出来。切换靠 `start.sh --params <名字>` 传给 launch 的 `params_file`。

> ⚠️ `line_follower_params.yaml` 里同时有 `line_follower_node`、`detection_turn_controller`、`cmd_mux_node` 三段，拆文件时**三段都要在**，否则那两个节点会退回代码默认值（见 R19）。

### 9.4 赛中锁死

| 状态 | 参数表单 | 重启按钮 | 立即停车 |
|---|---|---|---|
| 未发车 | 可编辑 | 可用 | 可用 |
| **已发车** | **置灰 + 后端 423** | **置灰 + 423** | **可用** |

判定依据：面板订阅到的 `line_follower/start_cmd` 为 true。

### 9.5 验收

- [ ] 改 `kp` 2.45→2.80 → 保存 → 重启巡线 → `ros2 param get /line_follower_node kp` 返回 2.80
- [ ] 发车后点保存 → 前端灰、后端 423
- [ ] 发车后点「立即停车」→ 车停，且 `start_toggle` 窗口里再按空格能重新发车

---

## 10. P4 — `[DEC]` 决策日志

### 10.1 格式

```
[DEC] <区域> <事件> k=v k=v ...
```

只打**状态跃迁**，不打每帧。例：

```
[DEC] yellow count=1/4 stop_begin t=6.42
[DEC] yellow count=1/4 stop_end   dur=1.03
[DEC] state tracking->lost
[DEC] cone side=L risk=0.42 area=0.0018 in_path=1
[DEC] digit best=6 conf=0.71 xdiff=+0.18 yaw=-0.29 dur=0.38
[DEC] mux active=detection
[DEC] safety deadman_on silence=0.42
```

### 10.2 埋点位置

| 文件 | 位置 | 事件 |
|---|---|---|
| `line_follower_node.cpp` | 黄线锁存处（`yellow_line_detection_count_++` 附近） | `yellow count=n/4 stop_begin` |
| 同上 | 停车计时结束处（`yellow_line_stop_active_=false` 附近） | `yellow stop_end dur=x` |
| 同上 | `permanent_stop_=true` 处 | `yellow final stop` |
| 同上 | `current_state` 变化处 | `state a->b` |
| 同上 | `update_segment()` 段号变化 | `seg change from=1 to=2 count=2/4 dist=0.42` |
| 同上 | 等里程边界的那一段（岔路段2 / 环岛段3）2 Hz | `seg d=1.20 cnt=3 thr=3.30`（校准 `seg3_start_dist_m` / `seg4_start_dist_m` 用） |
| 同上 | `soft_red_active` 由 false 变 true 处 | `cone side= risk= area= in_path=` |
| 同上 | 丢线进入/恢复处 | `lost_enter` / `lost_exit` |
| `detection_turn_controller.cpp` | `trigger_turn()` | `digit best= conf= xdiff= yaw= dur=` |
| `cmd_mux_node.cpp` | 三个 callback 里的优先级分支 | `mux active=stop\|detection\|follower`（仅切换时） |
| `safety_node.py` | 兜底开关 | `safety deadman_on/off` |
| `panel/server.py` | 写参数、重启、estop | `panel param_write` / `restart` / `estop` |

### 10.3 实现方式

C++ 侧统一用一个宏，避免手写前缀出错：

```cpp
#define DEC_LOG(logger, ...) RCLCPP_INFO(logger, "[DEC] " __VA_ARGS__)
```

### 10.4 CSV 汇总（D13）

`tools/dec_collector.py`：单写者，tail 本次运行目录下所有 `*.log`，把 `[DEC]` 行拆成结构化记录写入 `events.csv`。

```csv
ts_iso,t_rel,source,area,event,kv
2026-10-20T21:33:07.412,18.320,follow,yellow,stop_begin,count=2/4
2026-10-20T21:33:08.477,19.385,follow,yellow,stop_end,dur=1.03
2026-10-20T21:33:12.900,23.808,follow,cone,avoid_on,side=L risk=0.42
```

- `t_rel` = 相对本次 `start.sh` 启动的秒数，便于和视频时间轴对齐
- 只通过一个进程写，**不存在多进程写同一文件的问题**
- 面板的计时器与估算分也从同一份 CSV/事件流取数，保证两处一致

---

## 11. P5 — 计时器与估算分

### 11.1 计时器

| 时机 | 动作 |
|---|---|
| `line_follower/start_cmd` **上升沿**（false→true） | 开始计时 |
| 第 4 根黄线 `permanent_stop` 触发 | 停止计时 |

页面顶部用 **48 px 等宽大字**显示 `mm:ss.d`。

> 这一条直接顶替实体秒表，满足规则「参赛队需自备计时器并将计时器展示在镜头前」。

### 11.2 估算分

```
估算分 = elapsed + 15×未停稳 + 30×闯禁区 + 10×撞锥 + 15×压线 − 5×bonus 累计
```

数据来源：`[DEC]` 事件流。**这是估算，不是裁判判定**，面板上标注“估算”。

> 现阶段「闯禁区」「压线」还没有检测代码，先显示 `0`，等算法侧补上自动生效。

### 11.3 验收

- [ ] 按空格 → 计时器开始走
- [ ] 第 4 根黄线 → 计时器停住，数字冻结

---

## 12. 风险登记表

> 便于定位：每条都有 `ID / 现象 / 原因 / 怎么查 / 缓解`。实现时任何一条踩到，直接报 ID。

### R01 下位机无超时保护
- **现象**：Ctrl+C 或节点崩溃后车继续跑，速度不变
- **原因**：底盘固件没有指令超时看门狗，`src/rdkx5.md` 第 9 节已记录
- **怎么查**：`ros2 topic info /cmd_vel` 看是否还有发布者；`ros2 topic hz /cmd_vel` 是否停
- **缓解**：`safety_node` 兜底（P1）；任何时候手上备一条零速命令

### R02 `ros2 launch` 子节点死了父进程还活着
- **现象**：④ 窗口没新日志，但 `ps` 看进程在
- **原因**：`launch` 默认不因单个节点退出而终止
- **怎么查**：`ros2 node list` 里缺 `line_follower_node` / `detection_turn_controller` / `cmd_mux_node`
- **缓解**：守护用 `ros2 node list` 计数 == 3 判定，不用 `ps`

### R03 `DISPLAY` 丢失导致窗口弹不出来
- **现象**：`start.sh` 跑完什么都没有，或 `xterm: Can't open display`
- **原因**：在 `screen` / `nohup` 里执行；或 SSH 没开 X11 转发
- **怎么查**：`echo $DISPLAY` 应为 `localhost:10.0` 之类；`xauth list` 有记录
- **缓解**：文档写明必须在 MobaXterm 的 SSH 终端直接执行；`start.sh` 开头检查 `DISPLAY` 为空则直接报错退出

### R04 关掉 xterm 窗口的行为（已按 D12 定案）
- **现象**：关窗口后该进程不再恢复，`status.json` 里 `alive=false` 且 `restarts` 不变
- **原因**：关窗口 = SIGHUP = 退出码 129，按 D12 判为“主动停止”，不重启
- **怎么查**：`tools/logs/latest/<name>.exit` 内容为 `129`；`[DEC] proc_exit name=<x> code=129`
- **缓解**：这是**刻意行为**，你可以直接关窗口停单个进程。要重新拉起，用面板「重启」或 `./tools/start.sh --stop <name>` 后重跑
- **副作用（要知道）**：如果进程是被**外部 SIGKILL/SIGHUP 干掉**（不是关窗口），也会被当成 129 而不重启。区分办法见 `status.json` 的 `exit_code` 字段

### R05 面板发车与 `start_toggle_node` 冲突
- **现象**（如果做面板发车）：面板发 true，车动了 0.1 s 又停
- **原因**：`start_toggle_node` **10 Hz 持续发布**它自己的状态，会立刻把 true 覆盖成 false
- **怎么查**：`ros2 topic echo /line_follower/start_cmd` 能看到每秒 10 条 false
- **缓解**：**面板不做发车按钮**（已锁定 D2/非目标）。发车唯一入口是 ⑤ 窗口按空格。面板只显示 `started` 状态 + 提供 estop

### R06 `estop` 只发几帧会被 mux 恢复
- **现象**：点「立即停车」，车停一下又走
- **原因**：`cmd_mux_node` 的 `stop_timeout` 是 0.5 s，超时后回到低优先级
- **怎么查**：`ros2 param get /cmd_mux_node stop_timeout`
- **缓解**：estop 连续发 **30 帧 @20 Hz = 1.5 s**，远超 0.5 s；同时面板提示“如需持续停止请到 ⑤ 窗口按空格”

### R07 改参数不生效
- **现象**：面板显示新值，车行为不变
- **原因**：`declare_parameters()` 把参数读进成员变量（`kp_` `target_speed_` `max_yaw_rate_` 等），只在构造时执行一次
- **怎么查**：`ros2 param get /line_follower_node kp` 返回新值但车不变 → 就是这个
- **缓解**：**必须重启节点**才能生效（D5 已锁定）；面板在保存后弹提示「需重启巡线生效」

### R08 写回 YAML 丢注释
- **现象**：改一次参数，YAML 里原有注释全没了
- **原因**：`yaml.safe_dump` 不保留注释（`ruamel.yaml` 才行，未安装）
- **怎么查**：`git diff src/.../fork.yaml`
- **缓解**：写入前备份成 `.bak`；**参数说明搬到 `params_schema.py` 和面板上**，不依赖 YAML 注释；这也让面板更好用

### R09 面板抢 CPU 导致车抖
- **现象**：打开面板页面后巡线开始左右摆
- **原因**：MJPEG 编码 + X11/HTTP 占用；8 核但巡线是单线程 30 fps
- **怎么查**：`top` 看 `server.py` CPU；关掉页面看是否恢复
- **缓解**：限 **6 fps** + JPEG 质量 70；页面不显示时生成器必须退出；必要时把调试图发布频率也降下来

### R10 面板崩溃会怎样
- **现象**：浏览器打不开
- **原因**：`server.py` 异常
- **怎么查**：`tools/logs/latest/panel.log`
- **缓解**：面板**不在控制链路上**，崩了不影响车；`start.sh` 会重启它（最多 3 次）

### R11 端口 8080 被占
- **现象**：面板起不来，`Address already in use`
- **怎么查**：`ss -ltnp | grep 8080`
- **缓解**：`start.sh` 从 8080 往上试到 8089，选中的端口写进 `status.json`，日志里打出来

### R12 代理把本地流量也劫了
- **现象**：面板自己打不开、`ros2` 命令变慢、`localhost` 访问超时
- **原因**：`http_proxy` 影响所有 HTTP 客户端
- **怎么查**：`echo $http_proxy`；`curl -x "" http://127.0.0.1:8080`
- **缓解**：`no_proxy=127.0.0.1,localhost,10.151.31.0/24`（见 4.3）

### R13 丢线后继续前进（既有代码风险）
- **现象**：在赛道边界处车不停，直接冲出去
- **原因**：`line_lost_forward_speed: 0.3`，且没有“lost → stop”逻辑
- **怎么查**：日志 `[DEC] state tracking->lost` 之后车还在动
- **缓解**：**这是算法侧改动（P1 阶段）**，本工具链只负责把它**记录**下来（`[DEC] lost_enter`），不修

### R14 黄线停车计时从“检测到”开始（既有代码风险）
- **现象**：录像里车还在滑行，1 秒就过了，看着像“没停住”
- **原因**：`yellow_stop_duration_` 从 `yellow_line_detection_time_` 起算，不校验真实静止
- **怎么查**：`[DEC] yellow stop_begin` 与 `stop_end` 的 `dur` 约 1.0，但视频里车在动
- **缓解**：算法侧改动，工具链只记录 `dur`

### R15 红锥与禁行区混淆（既有代码风险）
- **现象**：车把 24×13 cm 的红色禁行框当路锥绕，或直接开进去
- **原因**：现有代码只分“横向红线 / 红锥”两类，没有“空心矩形”判据
- **怎么查**：`[DEC] cone ...` 里 `area` 很大但 `side` 与视觉不符
- **缓解**：算法侧改动；本阶段**记录事件**即可

### R16 `start.sh` 自己被 kill → 全部进程成孤儿
- **现象**：SSH 断开或 Ctrl+C 后，车还在跑，xterm 全没了
- **原因**：xterm 里的进程是 `start.sh` 的子进程，父进程死了它们不一定死
- **怎么查**：`ps aux | grep line_follower`
- **缓解**：`start.sh` 里 `trap` 处理 INT/TERM/HUP；**并且 `safety_node` 独立存活**（它不依赖 `start.sh` 的父子关系，靠 ROS 话题工作）→ 上游全死时它会把车停住

### R17 SSH 断线 → 5 个窗口全丢
- **现象**：网络抖一下，所有 xterm 消失
- **原因**：X11 客户端依赖 SSH 会话
- **怎么查**：重连后 `ps aux | grep ros2` 看还有没有残留
- **缓解**：日志 `tee` 落盘不丢；重连后重跑 `start.sh`。**不要**为了保活去套 `screen`（会导致 R03）

### R18 `tee` 缓冲导致日志延迟
- **现象**：面板里日志慢半拍
- **原因**：`tee` 走块缓冲
- **怎么查**：`tail -f` 直连文件看是否也慢
- **缓解**：Python 进程加 `-u`（无缓冲）；C++ 侧 `RCLCPP` 默认行缓冲，可接受

### R19 拆两套 YAML 时漏段
- **现象**：切到 `fork.yaml` 后，YOLO 转向或仲裁行为变成代码默认值
- **原因**：原 YAML 里有 `line_follower_node` / `detection_turn_controller` / `cmd_mux_node` **三段**，只搬一段就会漏
- **怎么查**：`ros2 param get /detection_turn_controller turn_duration` 看是不是 YAML 里的 0.38
- **缓解**：拆文件用脚本复制，拆完逐段 diff；验收清单里加一条

### R20 面板的 `rclpy` 订阅与主程序抢节点名
- **现象**：面板启动时报节点名冲突
- **原因**：用了和现有节点相同的名字
- **怎么查**：`ros2 node list` 出现重复
- **缓解**：面板节点固定命名 `panel_node` / `safety_node`，与现有不重名

### R21 分段切错（段号与车实际位置不符）
- **现象**：某段参数明显不对（比如还在环岛里就用上了段 4 的高速），面板段号与视频位置对不上
- **原因**：① 黄线少计/多计一根 → 段 2、3 整体错位；② `seg4_start_dist_m` 没按首跑校准；③ `track_mode` 与实际地图不符
- **怎么查**：`[DEC] seg change` 的时间戳与视频对齐；调试图上的 `SEG x/4 d=…`；`[DEC] param effective #0 mode=… segs=…`
- **缓解**：三个边界参数都在面板「分段切换」组可改；段号**单调递增**不会来回跳；`seg4` 默认偏晚；启动时若检测到旧键（`target_speed` 等）会打 `stale legacy keys` 警告
- **代价**：晚切约 0.2 s 圈速；早切可能切内线，所以宁可晚

---

## 13. 工期与验收节点

| 阶段 | 交付 | 工期 | 独立可用？ |
|---|---|---|---|
| P0 | `start.sh` + 5 窗口 + 日志 + 守护重启 + `status.json` | 1 天 | ✅ 有它就够日常跑 |
| P1 | `safety_node.py` | 0.5 天 | ✅ 单独提升安全性 |
| P2 | 面板只读（status + 2 路 MJPEG + 日志） | 1 天 | ✅ 能看 |
| P3 | 参数编辑 + 受控重启 + 发车后锁死 | 0.5 天 | ✅ 能调 |
| P4 | `[DEC]` 埋点 + 决策过滤 | 0.5 天 | ✅ 能追 |
| P5 | 计时器 + 估算分 | 0.5 天 | ✅ 能拍 |

合计 **4 天**。每阶段结束按对应「验收」小节逐条过，不通过不进下一阶段。

---

## 14. 回滚

全部新增文件在 `tools/` 下，**不改现有包结构**。`[DEC]` 埋点是对 3 个 cpp 的**纯增量**（只加日志，不改逻辑）。

回滚方式：

```bash
git checkout -- src/line_follow_ws/src/line_follower/src/   # 撤销埋点
rm -rf tools/                                               # 撤销整套工具
```

`tools/` 不进 `colcon` 构建，删掉不影响编译。

---

## 15.1 实施状态（v1.2，2026-10-20）

| 交付物 | 状态 | 验证方式 |
|---|---|---|
| `tools/start.sh` | ✅ 完成 | `--headless --no-panel` 实跑：5 进程按序拉起、日志落盘、`status.json` 更新、TERM 停止后残留 0 |
| `tools/wrap.sh` | ✅ 完成 | 同上，退出码正确写盘 |
| `tools/safety_node.py` | ✅ 完成 | 实测在全部上游静默 0.32s 后触发 `deadman_on` |
| `tools/dec_collector.py` | ✅ 完成 | 实测解析 5 类事件写入 `events.csv`，追加日志后增量续写正确 |
| `tools/panel/server.py` | ✅ 完成 | 实测 `/api/status`、`/api/params`、POST 写入、越界拦截、发车后 423、`/api/estop` |
| `tools/panel/index.html` | ✅ 完成 | 未在浏览器里做视觉验收（无显示器），接口层已通 |
| `tools/config/fork.yaml` / `roundabout.yaml` | ✅ 完成 | 与源 YAML 逐节点比对，108 个参数项完全一致 |
| `[DEC]` 埋点（3 个 cpp） | ✅ 完成 | 重编译通过，运行日志里可见 `[DEC]` 事件 |
| ④ 子节点崩溃自动重启 | ⚠️ **未做破坏性验证** | 检测逻辑确认在运行（每 5s 计数），但“杀掉子节点→自动重启”这一段没跑通测试（安全策略拦住了 kill 命令） |
| 面板发车后锁定 UI | ⚠️ 后端已验证，前端置灰未在浏览器确认 | 423 已实测 |

**首次实机建议顺序**：先 `./tools/start.sh`（5 窗口）→ 确认五个窗口都有输出 → 浏览器开 `localhost:8080` → 最后才按空格发车。

---

## 15. 已确认（v1.1 评审结论）

| # | 问题 | 结论 |
|---|---|---|
| 1 | 重启判据 | 仅退出码**非 0 且非 129** 才重启（D12） |
| 2 | `[DEC]` 是否落 CSV | **要**，汇总到 `tools/logs/run_*/events.csv`（D13） |
| 3 | 两套 YAML 命名 | **`fork.yaml` / `roundabout.yaml`**（D14） |

> 电池电压 `/PowerVoltage` 暂不显示（可后续在状态栏加一行，不影响其他设计）。
