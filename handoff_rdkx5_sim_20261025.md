# 交接文档：RDK X5 智能车 · 仿真与速度最优解

> 生成时间：2026-10-25（10-26 从 /tmp 迁回工作区）　工作区：`/home/sunrise/ws_rdkx5_1`
> 用途：新会话接手。读完这份即可继续，不需要重读整个仓库。

---

## 1. 一句话现状

仿真框架已搭好并跑通（赛道提取 / 相机渲染 / 世界节点 / 速度剖面 / 一键启动），
但**还没完成第一次闭环圈速验证**，因此"最优解"尚未经仿真确认。
下一步是扫恒速拿安全上限，再改控制器落地分段速度。

---

## 2. 背景与目标

- 赛事：2026AIC 智能车视觉规划赛，复赛 10-31 ~ 11-01 线上提交（每张地图只跑一次）。
- **复赛只有黄色停车线**，红锥 / 禁行区 / 数字标贴都是决赛内容。
- 计分：`时间 + 15×未停稳 + 30×闯禁区 + 10×撞锥 + 15×压线 − 5×Bonus`，三次取最优。
- 两张地图固定：`5-3m地图环岛.pdf`、`5-3m地图岔路S.pdf`（矢量 CAD，5 m × 3 m）。
- 用户目标：以最短时间完成，同时保鲁棒性。

规则细节见 `2026AIC智能车视觉规划赛_比赛规则提炼.md`；工具链设计见 `tools/DESIGN.md`。

---

## 3. 已完成

### 3.1 代码（全部在 `tools/sim/`，`src/` 一行未改）

| 文件 | 作用 | 状态 |
|---|---|---|
| `extract_track.py` | 矢量 PDF → 中心线/曲率/黄线弧长 + 鸟瞰 PNG | 跑通 |
| `render.py` | 鸟瞰 → 640×480 相机视角（remap，逐像素反投影） | 跑通 |
| `world_node.py` | 运动学 + 一阶滞后，发 `/sim/odom` `/image_raw` `/sim/gt` | 30 Hz 跑通 |
| `profile.py` | 曲率限速 + 前向后向扫描 → 时间最优速度剖面 | 出结果 |
| `run_sim.sh` | 一键闭环，mux 输出重定向到 `/sim/cmd_vel` | 待首次完整验证 |

产物：`tools/sim/tracks/{fork_s,roundabout}.{json,png}`、`tools/sim/profiles/*.json`。

### 3.2 Git（4 个 commit，工作区当时干净）

```
ba8505a 补全运行产物忽略规则
2bb96fd 运行产物不再纳入版本管理
d05fe02 新增工程全景文档：白色路面上的 8 个进程
398458a 调整巡线速度参数：岔路 0.26/0.3，环岛 0.3/0.3
```

`tools/sim/` 尚未提交。

### 3.3 实车数据（`/home/sunrise/bags/`）

| bag | 内容 | 用途 |
|---|---|---|
| `calib_static` | 18.3 s 静止，550 帧 | 相机标定（地上放 11 cm 盒子） |
| `step` | 首版阶跃（时序不干净，已弃用） | — |
| `step3` | 3 轮 × 4 个转向幅度 + 速度阶跃，71 s | 延迟/增益标定 |
| `lap_fork2` | 岔路整圈 54.5 s，4 根黄线正常 | **唯一完整真实圈** |
| `lap_fork` | 中途冲出赛道，无参考价值 | — |
| ~~`lap_roundabout2`~~ | 录制中断（6.69 GB，仅 20 s 有效），已删 | — |

---

## 4. 硬标定数据（已多次交叉验证，可直接用）

| 参数 | 值 | 置信度 |
|---|---|---|
| 相机高度 h | 110 mm | 用户实测 |
| 相机俯角 α | 12.5°（12–13） | 用户实测 |
| 像素焦距 f | 325 px | 由地平线落在 `roi_top=168` 反推，与盒子标定闭合 7% |
| 轴距 L | **139.5 mm** | 用户实测（URDF 写 213.5 mm，**不可用**） |
| `τ_speed` | 0.61 s | 3 轮 598/611/614 ms |
| `τ_steer` | 0.22–0.30 s | 随转角变化 |
| 速度增益 | 0.99 | 3 轮 |
| 舵角上限 | **21°** | 6 段一致，两算法互验 |
| 最小转弯半径 | **0.36 m** | 同上 |
| 左右对称性 | 对称（< 2%） | 开阔地重测，推翻早前"右转弱 15%"的误判 |
| 减速度 | ≈ 0.4 m/s² | 0.4→0 约 1 s，滑行 0.2 m |

**预瞄距离**（ROI 168–384）：0.15 m（底）– 1.18 m（顶），地平线正好落在 `roi_top`，
说明 `roi_top_ratio=0.65` 就是照这个相机角度调的，参数没问题。

**注意**：`lap_fork2` 全程电压 10.44 V（3S 满电 12.6 V），实际速度比指令低约 15%。

---

## 5. 赛道数据（矢量图提取）

| | 长度 | Rmin | 车道宽 | 黄线弧长 s (m) |
|---|---|---|---|---|
| 岔路 `fork_s` | 10.08 m | 0.34 m | 434 mm | 0.77 / 2.33 / 7.61 / 9.97 |
| 环岛 `roundabout` | 10.70 m | 0.40 m | 430 mm | 0.82 / 2.48 / 8.07 / 10.59 |

- 黄线由 odom 实测位置按比例映射到几何弧长（odom 11.07 m vs 几何 10.08 m，差 8%）。
- 规则表里的 130/20/30/6 cm **是四个独立位置（起点、两个岔口、终点前），不是间距**。
  按此解读，第 2、3 根与几何 2.33/7.61 几乎重合；只有第 1 根因发车点摆放偏前差 45 cm。
- 曲率做了 21 点（≈42 cm）中值滤波：滤前 Rmin 0.14 m（锯齿伪影），滤后 0.34/0.40 m，与车辆能力吻合。

---

## 6. 关键结论

1. **当前架构的天花板不是电机，是控制律。** `line_follower_node` 恒速 + 纯 P + 无 odom，不知道自己在赛道哪个位置。
2. **停车是最大固定成本。** 日志写"停 1.0 s"，odom 显示真正静止只有 0.35–0.50 s；3 次停车 + 加减速共约 6–9 s，占 38 s 的 17–23%。
3. **`max_yaw_rate=1.5` 在 0.4 m/s 时勉强够**（物理上限约 1.04–1.16），低速时完全达不到——这不是 bug，是舵角饱和。
4. **环岛几何余量只有 5–12 cm**：所需舵角 16°，能力 21°。
5. 理论速度剖面：岔路 20.5 s、环岛 21.2 s（停车 1.0 s）；停车改 0.35 s 后 18.5 / 19.2 s。
   **这是"完美跟踪"下界，实际要打折扣。**

---

## 7. 当前阻塞（最重要）

- AI 侧 `execute_command` 需要用户审批，最近多次超时被取消，**无法自动执行命令**。
  涉及启停进程、跑仿真、chmod 的操作必须由用户在终端手动执行，AI 只能改文件。
- 用户已手动运行过 `bash tools/sim/run_sim.sh fork_s`，但**首次因 `set -u` 报错**；
  修复后曾误驱动真车——原因是 launch 用 `LaunchConfiguration` 硬覆盖了
  `cmd_mux` 的 `output_topic`（yaml 不生效）。现已加三层保险，尚未确认用户端跑通。
- **首次闭环圈速尚未拿到**，模型可信度未知。

### 已修复的两个坑（新会话不要再踩）

1. `run_sim.sh` 不能用 `set -u`：`/opt/tros/humble/setup.bash` 会引用未定义的
   `AMENT_TRACE_SETUP_FILES`。
2. **必须显式传 `cmd_vel_topic:=/sim/cmd_vel`**：
   `launch/line_follower_with_detection.launch.py:107` 用 `cmd_vel_topic` 参数覆盖
   mux 的 `output_topic`，默认值是 `/cmd_vel`，不传就会驱动真底盘。

---

## 8. 下一步（按优先级）

### P0 完成首次闭环验证（约 10 分钟）

```bash
pkill -INT -f turn_on_wheeltec_robot     # 真底盘必须停
bash tools/sim/run_sim.sh fork_s
```

- 必须看到 `mux output 已确认: /sim/cmd_vel`，否则立即停。
- 用当前 0.3 恒速跑完，记录圈速（从 `/tmp/sim_follow.log` 的
  `[DEC] yellow stop_begin` 到 `final_stop`）。
- **判据**：落在 45–48 s → 模型可信；落在 36–40 s → 偏乐观 6–10 s，要标定
  `a_acc`/`a_dec`、补转弯掉速项。

### P1 扫恒速（约 5 分钟，不改代码）

依次跑 0.30 / 0.35 / 0.40 / 0.45 / 0.50，判据只有一条：**跑完 4 根黄线且不出界**。
目标是拿到当前 P 控制架构的**最大安全恒速**（预期 0.35–0.40）。
建议写 `tools/sim/sweep.py` 自动化：改参数 → 启动 → 等 `final_stop` → 记圈速 → 关机。

### P2 落地分段速度（约 1 天，需改 `src/`）

在 `line_follower_node.cpp` 提速计算处（`745:752`）加：

1. 订阅 `/odom` 积分里程 `s`，每过一根黄线把它重置到
   `tracks/*.json` 里的 `yellow_lines_s_m`（黄线是精确锚点，可消漂移）。
2. 查 `profiles/*.json` 的 `v_target` 替换固定 `target_speed`。
3. 保留视觉 P 做横向纠偏，不做前馈转向（6 天内风险太高）。

改完用仿真验证，再上实车 0.45。

### P3 待补数据

- 环岛完整一圈 bag（`lap_roundabout2` 废了，需重录）。录制命令见对话历史：
  `ros2 bag record -o ~/bags/lap_roundabout2 --max-cache-size 400000000 ...`
- 第 1 根黄线距发车线的准确距离（现场卷尺量）。

---

## 9. 文件索引

```
tools/sim/
├── extract_track.py       PDF → 赛道 JSON/PNG
├── render.py              CameraModel / TrackRenderer（含预瞄距离表）
├── world_node.py          仿真世界节点（默认 /sim/cmd_vel /sim/odom）
├── profile.py             时间最优速度剖面
├── run_sim.sh             一键闭环
├── tracks/*.json|png      两张地图的几何
└── profiles/*.json        速度剖面（vmax 0.8 / 1.0 各一份）

tools/config/fork.yaml         target_speed 0.26 / max_speed 0.3
tools/config/roundabout.yaml   target_speed 0.3  / max_speed 0.3
tools/panel/                   Web 面板（赛中只读）
tools/DESIGN.md                工具链设计 + 风险登记表 R01–R20

src/line_follow_ws/src/line_follower/src/
├── line_follower_node.cpp     巡线主体（速度公式 745-772，停车 717-738，丢线 773-794）
├── cmd_mux_node.cpp           仲裁（output_topic 默认 /cmd_vel）
├── start_toggle_node.cpp      空格发车，10 Hz 持续发 false
└── launch/line_follower_with_detection.launch.py   ← 第 107 行是 output_topic 覆盖点
```

---

## 10. 用户偏好

- 中文回答，**简洁、直接给命令**，先给结论再给解释。
- 不要长篇铺陈；表格化优先。
- 涉及启停进程 / 跑仿真，**给命令让用户自己执行**，不要尝试自动执行。
- 真车安全第一：任何可能发 `/cmd_vel` 的操作必须先确认真底盘已停。

---

## 11. 建议 skills

| skill | 何时用 |
|---|---|
| `think` | P2 改控制器前做方案评审（需要先确认罚时/鲁棒性的取舍） |
| `handoff` | 再次会话中断时压缩上下文 |
| `ponytail` | P2 落地时要求最小改动，避免过度设计 |
