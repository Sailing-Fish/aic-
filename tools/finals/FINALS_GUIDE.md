# AIC 2026 智能车视觉规划赛 · 全国决赛实战手册

> 本手册面向 RDK X5 + Wheeltec `mini_akm` 实车（280×191×140 mm，轴距 142 mm，前防撞条，2.8 mm 无畸变 105° USB 相机）。所有决赛模块均采用**解耦可切换开关**，随时可一键回退到复赛基线。

---

## 1. RDK X5 拉取、编译与一键启动

### 1.1 编译工作区
```bash
cd ~/aic--main   # 或你的工作区根目录
source /opt/tros/humble/setup.bash

# 1) 若改动了 car_ws (ekf.yaml / base_serial.launch.py)
cd src/car_ws && colcon build --symlink-install && cd ../..

# 2) 编译 line_follow_ws（C++ 巡线 + 走廊规划 + 仲裁）
cd src/line_follow_ws
colcon build --packages-select line_follower --cmake-args -DCMAKE_BUILD_TYPE=Release
cd ../..

# 3) 刷新环境
source src/car_ws/install/setup.bash
source src/line_follow_ws/install/setup.bash
```

### 1.2 启动决赛配置 vs 复赛基线
```bash
# 决赛模式（推荐）：启用 relative_border 抗光照 + 走廊动态避障 + FinishGuard + 反拖急停
./tools/start.sh --params finals_fork
./tools/start.sh --params finals_roundabout

# 复赛基线回退（零风险对照）
./tools/start.sh --params fork
./tools/start.sh --params roundabout
```
- Web 面板通过 SSH 隧道访问 `http://127.0.0.1:8080/`，顶部实时显示 **`/PowerVoltage` 电池电压**（低于 **11.6V** 红色告警并弹窗，请立刻换电）。

---

## 2. 核心解耦模式开关速查

配置文件位置：`tools/config/`（`fork.yaml`、`roundabout.yaml`、`finals_fork.yaml`、`finals_roundabout.yaml`）。所有开关均可在 Web 面板中修改并点击 **「保存并重启巡线」** 生效。

| 参数名 | 可选值 | 作用说明 | 复赛基线 | 决赛推荐 |
|---|---|---|---|---|
| `planner_mode` | `legacy_segment` / `finals_corridor` | `legacy_segment` 沿用黄线计数分段；`finals_corridor` 启用动态可行走廊 + 障碍记忆 + 岔路静态投票 + FinishGuard | `legacy_segment` | `finals_corridor` |
| `lane_seg_mode` | `legacy_hsv` / `relative_border` / `hybrid` | `relative_border` 用蓝垫色差 + 局部黑边 + 帧内 Lab 动态白参考，抗老化/眩光/色温漂移（零手动标定） | `legacy_hsv` | `relative_border`（实测后可切 `hybrid`） |
| `yellow_detect_mode` | `legacy_area` / `geometric_cv` / `yolo_cv_hybrid` | `geometric_cv` 要求横向细长几何特征并提取精确底边像素行；11 类 YOLO 就绪后切 `yolo_cv_hybrid` | `legacy_area` | `geometric_cv` → `yolo_cv_hybrid` |
| `enable_semantic_hole_fill` | `true` / `false` | 用检测框在白道二值图上回填白色，防止彩色/涂鸦 Bonus 数字卡片把赛道“挖洞”导致质心乱跳 | `false` | `true` |
| `enable_obstacle_memory` | `true` / `false` | 在 `/odom_combined` 维护红锥/禁行区世界坐标记忆，保留车头后方 `[-0.30m, 0]` 侧身障碍，防止后轮切内蹭 +30s 禁行区 | `false` | `true` |
| `allow_dual_digit_cover` | `true` / `false` | 直道双数字（间距 10cm）时：`false` 偏向大数字一侧避开小数字；`true` 允许同时覆盖（以规则现场解释为准） | `false` | `false` |
| `use_homography` | `true` / `false` | `true` 使用 4 点标定的 3×3 `homography_matrix` 做毫米级地面投影；`false` 自动退回实测针孔模型 | `false` | 标定后 `true` |
| `enable_detection_override` (`cmd_mux_node`) | `true` / `false` | 决赛统一由 `finals_corridor` 在同一走廊里协同避障与吃数字，必须设为 `false` 防止 `detection_turn_controller` 覆盖红障避让！ | `true` | `false` |
| `ae_awb_mode` (`camera.py`) | `auto` / `lock_on_start` / `manual_fixed` | `lock_on_start` 赛前自动适应现场光线，发车上升沿锁定曝光与白平衡，消除赛中亮度骤变 | `auto` | `lock_on_start` |

> **紧急兜底**：如果极端场馆光照下 `relative_border` 出现异常，可直接在 Web 面板切回 `legacy_hsv` 并微调「黄色停车线」「红锥与红线」「视觉阈值」下的 HSV 参数后重启巡线。

---

## 3. 三步离线 CV A/B 实验指南（到场 5 分钟定最佳分割模式）

不要凭感觉调参！到新场地后只需录 2–3 段 15 秒短片即可科学选出最优模式：

```bash
# 步骤 1：打开 50Hz 抗频闪，对比自动曝光与手动限曝光录制（推车走一圈含眩光/阴影/数字区）
python3 tools/finals/record_and_compare_cv.py --record \
  --device /dev/video0 --duration 15 --power-line-50hz \
  --exposure 0 80 150 --out tools/logs/venue_clips

# 步骤 2：离线运行 legacy_hsv vs relative_border vs hybrid 三模式对比
python3 tools/finals/record_and_compare_cv.py --compare tools/logs/venue_clips/clip_exp80.mp4 \
  --out tools/logs/cv_ab_exp80

# 步骤 3：查看终端对比表与三联对比视频 tools/logs/cv_ab_exp80/compare_side_by_side.mp4
# 选「丢线率最低」且「抖动 P95 最小」的模式填入 YAML 的 lane_seg_mode
```

---

## 4. 10 分钟 IPM 地面单应标定（零压黄线 + 精准障碍宽度）

坐标系原点固定在**车头泡沫防撞条正中心地面投影**，`+d` 向前（米），`+y` 向左（米）。

1. **摆点**：把车端正摆在 430 mm 直道中央，用卷尺在车头正前方 `0.30 m`、`0.60 m`、`1.00 m` 处标记左右黑线内缘（`y = +0.215 m` / `-0.215 m`，共 6 个点，至少 4 点）。
2. **抓拍并点选标定**：
   ```bash
   python3 tools/finals/calibrate_homography.py \
     --device /dev/video0 --save-snap tools/logs/ipm_snap.png \
     --save-points tools/logs/ipm_pts.json --out tools/logs/ipm_homography.yaml
   ```
   （无图形界面时，把 `ipm_snap.png` 拷到电脑查看像素坐标写入 JSON，再跑 `--points ipm_pts.json`。）
3. **验收标准**：终端报告中**平均地面误差 < 8 mm、最大误差 < 15 mm** 且矩阵可逆。
4. **写入配置**：将输出的 `use_homography: true` 与 `homography_matrix: [...]` 粘贴到 `finals_fork.yaml` / `finals_roundabout.yaml` 的 `line_follower_node` 下。
5. **顺手标定里程计比例**：用 `python3 tools/speed_step.py --speeds 0.45 --reps 2 --ask-tape` 测出 `odom/卷尺` 比例，填入 `odom_scale_factor` 或底盘启动参数 `odom_x_scale`。

---

## 5. 11 类 YOLO 合成增强、训练与 RDK X5 BPU 转换清单

类别顺序严格对应 `src/yolov8/mod1/out_finals11.list`：
`0..7: 1..8` | `8: red_cone` | `9: red_no_entry` | `10: yellow_stop_line`

### 5.1 合成与域随机化数据生成
```bash
# 建议先收集 50-150 张真实赛场无道具/有道具底图放到 data/bg/，实拍抠图放到 data/props/<类名>/
python3 tools/finals/synth_finals_dataset.py \
  --out data/finals11_ds \
  --num-train 2000 --num-val 300 \
  --bg-dir data/bg --props-dir data/props \
  --max-rot-deg 15 --calib-count 150
```
- 脚本**严禁水平/垂直翻转**并硬限制旋转角 `±15°`，自动生成彩色/条纹/棋盘干扰底数字卡片、10–15 cm 双数字对、空心 24×13 cm 红框、眩光与色温偏移，同时输出 `dataset.yaml` 和 `calib_list.txt`。

### 5.2 Ultralytics YOLOv8 训练关键超参（必检！）
```bash
yolo detect train model=yolov8n.pt data=data/finals11_ds/dataset.yaml \
  imgsz=640 epochs=120 batch=32 \
  fliplr=0.0 flipud=0.0 degrees=15.0 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 mosaic=0.5
```
> **红线警告**：训练参数中必须显式写死 `fliplr=0.0 flipud=0.0`，否则默认 `fliplr=0.5` 会把数字 `2/3/4/5/6/7` 镜像翻转导致识别崩溃！

### 5.3 RDK X5 Horizon OE 工具链 INT8 BPU 转换核对
1. 导出解耦 6 头 ONNX（与现用 `best_modified_yolov8.bin` 相同 3 尺度 `cls + dfl_reg` 结构，输出通道 `11` 和 `64`）。
2. 在 Horizon OpenExplorer YAML 中填入：
   - `cal_data_dir` / 校准集使用 `data/finals11_ds/calib_list.txt`（100–200 张真实+合成混合图，**切勿只用纯白底图做 PTQ 校准**）。
   - `input_type_rt: nv12`，`input_type_train: rgb`，`scale_value: 0.003921568627`。
3. 产物 `.bin` 放入 `src/yolov8/mod1/`，启动 `bpu_yolov8_node.py` 时指定：
   - `model_path` 指向新 11 类 `.bin`
   - `cls_list_path` 指向 `src/yolov8/mod1/out_finals11.list`
   - `class_agnostic_nms: false`（按类独立 NMS，保证 10cm 双数字互不抑制）
   - `class_conf_overrides: '{"red_no_entry": 0.25, "red_cone": 0.30, "yellow_stop_line": 0.35}'`（+30s 禁行区宁可偏灵敏）
- **平滑过渡**：在 11 类模型练好前，可直接继续跑旧 8 类 `.bin` + `out.list`，`bpu_yolov8_node.py` 自动按类别数兼容；需注意 `finals_corridor` 模式下黄线可由 `geometric_cv` 接管，而红锥/红禁行区避障依赖 11 类 YOLO 检测。过渡期若赛道有红障，请切为 `planner_mode: legacy_segment` + `lane_seg_mode: relative_border`（启用 HSV `detect_red()` 兜底），11 类 `.bin` 就绪后再切回 `planner_mode: finals_corridor`。

---

## 6. 决赛三轮实战策略与赛前检查

决赛每张图共 **3 次机会，取最优净成绩**（基础用时 + 罚时 - 5×Bonus，单轮限 10 分钟，卡滞 30 秒判停）。

### 6.1 三轮节奏建议
- **第 1 轮（保底完赛轮，零大罚时优先）**：
  - 速度压在 `0.38–0.44 m/s`，打开 `finals_corridor` + `enable_obstacle_memory: true` + `yellow_require_odom_still: true`。
  - 目标：**零压线（+15s/s）、零碰禁行区（+30s）、零黄线不停（+15s）、零撞锥（+10s）**，稳吃顺路高分 Bonus，先锁定一份有效且具竞争力的保底成绩，避免 3 轮无效触发 `2×最差有效成绩` 惩罚！
- **第 2 轮（提速 + 收益优化轮）**：
  - 根据第 1 轮 `[DEC]` 日志（`tools/logs/latest/events.csv`）核对每个弯道与黄线刹车余量，将直道巡航提至 `0.48–0.55 m/s`，确认反拖刹车（`yellow_brake_speed`）无倒车过冲，岔路充分触发静态投票选高分分支。
- **第 3 轮（冲刺轮）**：
  - 已有两轮有效成绩兜底时，按现场赛道实况微调 `target_speed` 与 `yellow_pre_brake_dist_m`，挑战极限净成绩。

### 6.2 赛前 60 秒快速检查清单
1. **电量**：Web 面板电压 `>= 12.0V`（低于 `11.6V` 立即换满电电池）。
2. **相机**：镜头擦净、支架拧紧无松动；`ae_awb_mode=lock_on_start` + `power_line_frequency=1`（50Hz）。
3. **安全链**：确认 `cmd_mux_node` 的 `enable_detection_override=false`（决赛模式），`/cmd_mux/heartbeat` 正常，急停按钮响应正常。
4. **仿真快测（改参后可选）**：
   ```bash
   python3 tools/sim/world_node.py --track tools/sim/tracks/fork_s.json \
     --odom-topic /odom --gain 0.85 --color-cast warm --glare 0.35 \
     --random-yellow 2 --cones 2 --no-entry 1
   ```
5. **合规**：发车后 Web 面板自动锁定（HTTP 423），严禁赛中远程干预；技术答辩前准备好实测 `cv_ab` 对比图、IPM 误差表与三轮 `[DEC]` 事件日志。
