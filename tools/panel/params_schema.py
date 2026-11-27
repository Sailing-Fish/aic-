#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面板参数表单的 schema：按“正交层级 (tier)”组织，聚焦全国决赛两页式调参。

两大层级（前端渲染为 2 个独立标签页）：
    1. finals_core  — ★ 决赛核心调参（现场只看这一页：19 个正交物理参数，零重叠）
    2. calibration  — ⚙ 开关与标定（模式切换、B1/B3/B5/B10 标定与应急覆盖、看门狗）

类型约定：
    float | int | bool | enum
    - 数值/布尔项：(key, label, type, min, max, step, help)
    - 枚举下拉项：(key, label, "enum", options_list, None, None, help)

支持地图：
    finals_fork / finals_roundabout
"""

TIERS = [
    {
        "id": "finals_core",
        "label": "★ 决赛核心",
        "subtitle": "日常只调这 4 组正交物理量：速度、刹车留白、避障余量、终点里程互不打架",
    },
    {
        "id": "calibration",
        "label": "⚙ 开关与标定",
        "subtitle": "模式切换、B1/B3/B5 几何标定、B10 红障 HSV 应急覆盖与看门狗",
    },
]


SCHEMA = [
    # =====================================================================
    # TIER 1: ★ 决赛核心调参（严格正交，日常调试只需关注这 4 组）
    # =====================================================================
    {
        "tier": "finals_core",
        "section": "line_follower_node",
        "group": "① 巡线速度与转向（基准速 × 弯道比例，零暗限速）",
        "desc": "直道快慢只调「巡航速度」；过弯稳不稳只调「弯道降速比例」和「横向刚度/阻尼」",
        "default_open": True,
        "items": [
            ("target_speed", "巡航速度 (m/s)", "float", 0.10, 1.00, 0.01,
             "唯一总速度基准（Run 1 保底 0.40–0.44，Run 2/3 可逐步提速）；不影响黄线停车点"),
            ("speed_gain_damping", "弯道自适应降速比例", "float", 0.00, 0.90, 0.05,
             "0=弯道不降速，0.5=满弯降一半；与巡航速度成比例正交，不再与事件限速叠乘"),
            ("corridor_kp_y", "走廊横向刚度 (rad/s/m)", "float", 0.5, 8.0, 0.1,
             "↑更紧贴目标轨迹（避障贴边更准），过大直道摆头；↓更柔顺但大弯慢半拍"),
            ("corridor_heading_gain", "航向阻尼增益", "float", 0.0, 3.0, 0.05,
             "抑制过弯回正与绕障后的摆头震荡"),
            ("max_yaw_rate", "最大转向角速度 (rad/s)", "float", 0.4, 3.0, 0.05,
             "转向硬上限；环岛建议 1.8–2.0，岔路 S 建议 1.5–1.7"),
        ],
    },
    {
        "tier": "finals_core",
        "section": "line_follower_node",
        "group": "② 黄线精准停稳（已彻底与 ROI / 巡航速解耦）",
        "desc": "停车位置只看「车头留白距离」；系统会自动先降到爬行速再按物理刹车公式停稳",
        "default_open": True,
        "items": [
            ("yellow_bumper_margin_m", "车头留白距离 (m)", "float", 0.02, 0.25, 0.01,
             "★ 唯一调停车前后位置的参数！怕车头压黄线 (+15s) 就调大，停太远就调小（推荐 0.08m）"),
            ("yellow_creep_speed_mps", "接近黄线爬行速 (m/s)", "float", 0.08, 0.45, 0.01,
             "进入预减速区后统一降到该速度再刹停，保证巡航提速后停点依然精准"),
            ("yellow_pre_brake_dist_m", "提前减速距离 (m)", "float", 0.25, 1.50, 0.05,
             "车头距黄线小于该距离开始降到爬行速（巡航 >0.55 m/s 时建议加大到 0.85m）"),
            ("yellow_stop_min_time", "最短停稳时长 (s)", "float", 0.20, 2.00, 0.05,
             "停稳后保持的最短时间；不得低于 0.30s"),
        ],
    },
    {
        "tier": "finals_core",
        "section": "line_follower_node",
        "group": "③ 避障余量与 Bonus 策略（全部为独立物理上限/米数）",
        "desc": "事件限速只作用于对应障碍物出现时；余量单位均为真实地面米数",
        "default_open": True,
        "items": [
            ("obstacle_speed_mps", "红锥绕行限速 (m/s)", "float", 0.08, 0.60, 0.01,
             "视野/记忆内有红锥时的独立速度上限（不影响无障碍直道）"),
            ("no_entry_speed_mps", "禁行区绕行限速 (m/s)", "float", 0.08, 0.50, 0.01,
             "绕行 +30s 红禁行区时的独立速度上限（宁慢勿碰）"),
            ("cone_margin_m", "红锥侧向留白 (m)", "float", 0.00, 0.08, 0.005,
             "车身外缘到红锥的最小物理间距（默认 0.025m）"),
            ("no_entry_margin_m", "禁行区侧向留白 (m)", "float", 0.00, 0.08, 0.005,
             "车身外缘到 +30s 空心红禁行区的最小物理间距（默认 0.030m）"),
            ("border_margin_m", "车道黑边留白 (m)", "float", 0.00, 0.08, 0.005,
             "车身外缘到黑线的保护间距（防压边 +15s/s，默认 0.025m）"),
            ("digit_min_conf", "吃数字置信度门槛", "float", 0.20, 0.95, 0.05,
             "Run 1 保底可拉高到 0.85 少冒险；Run 2/3 放回 0.55 吃 Bonus (-5s/个)"),
            ("max_digit_lateral_shift_m", "吃数字最大横移 (m)", "float", 0.00, 0.20, 0.01,
             "直道为吃数字允许偏离中线的最大距离；设为 0 = 本轮关闭直道吃数字"),
            ("digit_approach_speed_mps", "吃数字接近限速 (m/s)", "float", 0.08, 0.60, 0.01,
             "直道对准高分数字时的独立速度上限"),
        ],
    },
    {
        "tier": "finals_core",
        "section": "line_follower_node",
        "group": "④ 终点门控（按地图总长设好即冻结）",
        "desc": "防止决赛随机中途黄线误判为终点提前停赛",
        "default_open": False,
        "items": [
            ("finish_min_dist_m", "终点门控最短里程 (m)", "float", 4.0, 15.0, 0.1,
             "累计里程未达该值前，任何黄线只做临时停车不判终点"),
            ("finish_force_dist_m", "终点强判兜底里程 (m)", "float", 6.0, 18.0, 0.1,
             "超过该里程遇到黄线无条件判终点（即使视觉封口误漏）"),
        ],
    },

    # =====================================================================
    # TIER 2: ⚙ 开关与标定（模式开关、B1/B3/B5 标定、B10 红障应急、看门狗）
    # =====================================================================
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "Tier 1 增强功能开关（安全盾 / 摩擦自适应 / 风险决策）",
        "desc": "高频 A/B 对比开关：3-圆盘 DCBF 默认关闭，基础跑稳后再一键开启对比；γ 越小越保守",
        "default_open": True,
        "items": [
            ("enable_cbf_shield", "启用 3-圆盘阿克曼 DCBF 安全盾", "bool", 0, 1, 1,
             "默认 false 供实车安全 A/B 对比；开启后在转向/速度层硬拦截碰红框与车尾出界"),
            ("cbf_gamma", "DCBF 保守程度 γ（仅开盾生效）", "float", 0.10, 1.00, 0.05,
             "约束 h(k+1) >= (1-γ)h(k)；越小越提前温柔介入，越大越允许贴近（默认 0.60）"),
            ("enable_brake_adaptation", "启用黄线 1 参摩擦自适应 (EMA)", "bool", 0, 1, 1,
             "仅在黄线真正停稳后反算减速度微调摩擦系数（严钳位 [0.65, 1.25]）"),
            ("enable_risk_aware_decision", "启用风险感知 Bonus/岔路决策", "bool", 0, 1, 1,
             "岔路按置信一致度折减，直道按净空与偏移综合评分（不破坏硬安全红线）"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "模式总开关与 B10 白道 CV 红障兜底",
        "default_open": True,
        "items": [
            ("bev_planner_mode", "核心规划架构模式（一键灰度切换）", "enum",
             ["bev_dwa", "legacy"], None, None,
             "bev_dwa=第一性原理 BEV 米制占据栅格+拓扑指南针+阿克曼 DWA（推荐）；legacy=旧版走廊规划一键回退"),
            ("bev_imu_yaw_kp", "BEV IMU 陀螺仪内环增益", "float", 0.00, 1.20, 0.05,
             "补偿大角转向 20% 衰减与舵机机械中位偏置（默认 0.35；0=关闭内环）"),
            ("brake_min_cmd_mps", "黄线反拖死区下限 (m/s)", "float", 0.00, 0.40, 0.02,
             "越过 Wheeltec 电机 ~0.09 m/s 静摩擦死区，杜绝黄线刹车 timeout 滑行（默认 0.20）"),
            ("lane_seg_mode", "车道视觉分割模式", "enum",
             ["relative_border", "hybrid", "legacy_hsv"], None, None,
             "relative_border 抗光照/泛黄/眩光（零手动标定）；hybrid 结合宽松 HSV；legacy_hsv 极端应急"),
            ("yellow_detect_mode", "黄线检测模式", "enum",
             ["geometric_cv", "yolo_cv_hybrid"], None, None,
             "geometric_cv 横向几何+底边精准测距；11类 YOLO 就绪后可切 yolo_cv_hybrid"),
            ("enable_semantic_hole_fill", "启用语义回填(防数字挖洞)", "bool", 0, 1, 1,
             "把检测到的彩色/涂鸦数字区域回填白色，防止赛道二值图断裂"),
            ("enable_obstacle_memory", "启用障碍物世界坐标记忆", "bool", 0, 1, 1,
             "在 /odom_combined 保留车侧后轮区间障碍，防后轮切内蹭 +30s 禁行区"),
            ("allow_dual_digit_cover", "允许直道双数字同压", "bool", 0, 1, 1,
             "关=直道 10cm 双数字时偏向大数字避开小数字；开=允许同时覆盖"),
            ("use_homography", "启用 4 点 IPM 单应矩阵", "bool", 0, 1, 1,
             "true 使用 calibrate_homography.py 标定的矩阵；false 用针孔模型回退"),
            ("yellow_require_odom_still", "黄线严格要求里程计停稳", "bool", 0, 1, 1,
             "决赛推荐 true：未停稳前不提前放行，杜绝 +15s 未停稳罚时"),
            ("lost_line_safe_mode", "丢线安全减速模式", "bool", 0, 1, 1,
             "决赛推荐 true：丢线迅速减速并短时停机，防冲出赛道触发 +15s/s"),
            ("enable_cv_red_fallback", "启用白道 CV 红障兜底 (YOLO||CV)", "bool", 0, 1, 1,
             "开：YOLO 或白道内 CV 任一检到红障即避让（8类 YOLO 期必开，11类后仍作补漏）"),
            ("red_h_low_1", "B10 红色段1 H 下限", "int", 0, 30, 1, "CV 红障兜底默认 0"),
            ("red_h_high_1", "B10 红色段1 H 上限", "int", 0, 30, 1, "CV 红障兜底默认 10"),
            ("red_h_low_2", "B10 红色段2 H 下限", "int", 140, 180, 1, "CV 红障兜底默认 160"),
            ("red_h_high_2", "B10 红色段2 H 上限", "int", 140, 180, 1, "CV 红障兜底默认 180"),
            ("red_s_min", "B10 红色 S 下限 (0-255)", "int", 0, 255, 1, "反光发白可调低至 70"),
            ("red_v_min", "B10 红色 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低至 50"),
            ("red_cone_area_ratio_threshold", "B10 红障最小面积占比", "float", 0.0, 0.01, 0.0001,
             "单个红色轮廓占白道 ROI 面积比例门槛（默认 0.0006）"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "底盘、刹车与避障视野标定（B3 / B5，实测一次即冻结）",
        "desc": "由 speed_step.py / brake_probe.py 一次测准，与后轴几何/避障前瞻一起冻结",
        "default_open": False,
        "items": [
            ("odom_scale_factor", "里程计卷尺标定比例", "float", 0.70, 1.30, 0.005,
             "实测卷尺距离 / 里程计读数；修正终点门控与障碍记忆漂移"),
            ("rear_axle_to_bumper_m", "B3 后轴到前防撞条距离 (m)", "float", 0.05, 0.40, 0.01,
             "用于围绕后轴(EKF pose)旋转障碍物世界坐标，消除急弯车尾盲区旋转漂移（默认 0.18m）"),
            ("obstacle_steer_horizon_m", "B5 走廊避障前瞻距离 (m)", "float", 0.25, 1.20, 0.05,
             "走廊只优先响应前瞻地平线内的障碍，避免远处障碍过早偏离（默认 0.65m）"),
            ("obstacle_curve_min_horizon_m", "B5 急弯最小避障前瞻 (m)", "float", 0.20, 1.00, 0.05,
             "弯道内动态收紧的避障前瞻下限（默认 0.45m）"),
            ("obstacle_stagger_gap_m", "B5 错位双障纵向净距 (m)", "float", 0.05, 0.60, 0.05,
             "左右错位双障碍纵向间距大于该值时先绕近处再绕远处（默认 0.25m）"),
            ("yellow_brake_decel_mps2", "实测刹车减速度 (m/s²)", "float", 0.3, 5.0, 0.1,
             "用于物理公式 v²/(2a) 计算刹车触发距离；由 brake_probe.py 测得"),
            ("yellow_final_ramp_dist_m", "Tier 1 黄线近线斜坡减速距离 (m)", "float", 0.10, 0.80, 0.05,
             "距黄线小于该距离后线性柔降接近速度帽，压低停点误差至 0.5cm 级（默认 0.35m）"),
            ("yellow_final_creep_speed_mps", "Tier 1 黄线末段爬行速 (m/s)", "float", 0.08, 0.35, 0.01,
             "近线斜坡末段的最低接近速度帽（默认 0.18 m/s）"),
            ("voltage_decel_slope", "Tier 1 电压减速度前馈斜率 ((m/s²)/V)", "float", -1.0, 1.0, 0.02,
             "以 12V 为基准补偿电池电压波动对刹停减速度的影响；0.0 = 中性关闭"),
            ("yellow_control_latency_s", "视觉+控制总延迟 (s)", "float", 0.02, 0.35, 0.01,
             "补偿帧处理与串口延迟 v·τ（默认 0.10s）"),
            ("yellow_brake_speed", "反拖刹车速度 (m/s)", "float", 0.0, 1.5, 0.05,
             "触发停稳时发的负速度大小；0 = 关闭（滑停）。需先用 brake_probe.py 确认固件方向"),
            ("yellow_brake_release_speed", "反拖撤回阈值 (m/s)", "float", 0.0, 0.4, 0.01,
             "odom 速度降到该值以下立即撤回负速度，防止顶成倒车"),
            ("yellow_brake_max_time_s", "最长反拖时长 (s)", "float", 0.0, 3.0, 0.05,
             "反拖兜底超时，到点无条件撤回"),
            ("yellow_brake_period_s", "反拖点刹周期 (s)", "float", 0.0, 1.0, 0.02,
             "0 = 每帧连续发"),
            ("yellow_brake_pulse_s", "每周期反拖时长 (s)", "float", 0.0, 1.0, 0.02,
             "0 或 ≥周期 = 连续反拖"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "Tier 1 底层标定与硬件常量（标完即冻结，日常勿改）",
        "desc": "阿克曼底盘物理常量与离线 calibrate_crc.py 标定值；默认全部安全中性",
        "default_open": False,
        "items": [
            ("cbf_wheelbase_m", "DCBF 阿克曼轴距 (m)", "float", 0.08, 0.30, 0.002,
             "底盘前后轴距 L（硬件固定 0.142m，非换底盘勿改）"),
            ("cbf_max_steer_deg", "DCBF 最大前轮转角 (°)", "float", 15.0, 55.0, 1.0,
             "物理舵机转角硬上限（默认 35.0°）"),
            ("cbf_horizon_steps", "DCBF 前滚预测步数", "int", 2, 12, 1,
             "多步阿克曼弧段前滚步数（默认 6 步，勿随意拉大以免 CPU 波动）"),
            ("cbf_step_dt_sec", "DCBF 单步前滚步长 (s)", "float", 0.02, 0.15, 0.01,
             "单步前滚时间间隔（默认 0.06s）"),
            ("crc_no_entry_pad_y_px", "CRC 红禁行区框底外扩 (px)", "float", 0.0, 40.0, 0.5,
             "由 calibrate_crc.py 标定：向下外扩 v_bot 保守估计近距（默认 0.0 中性）"),
            ("crc_cone_pad_y_px", "CRC 红锥框底外扩 (px)", "float", 0.0, 40.0, 0.5,
             "由 calibrate_crc.py 标定：向下外扩 v_bot 保守估计近距（默认 0.0 中性）"),
            ("crc_yellow_pad_y_px", "CRC 黄线框底外扩 (px)", "float", 0.0, 40.0, 0.5,
             "由 calibrate_crc.py 标定：向下外扩 v_bot 防止低估黄线距离（默认 0.0 中性）"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "抗光照视觉微调（B1 / relative_border / geometric_cv）",
        "desc": "仅在换新场地出现明显白道漏检/蓝垫误检/黄线底边裁切问题时微调",
        "default_open": False,
        "items": [
            ("yellow_roi_bottom_ratio", "B1 黄线独立 ROI 底边裁切", "float", 0.0, 0.40, 0.01,
             "0.0=黄线检到画面最底行，完全解耦巡线 roi_bottom_ratio，确保近处黄线不断触"),
            ("blue_margin", "蓝垫色差抑制阈值", "int", 5, 80, 1,
             "B - max(R,G) > 该值判为蓝垫背景"),
            ("lab_blue_b_max", "Lab b 通道蓝垫上限", "int", 80, 130, 1,
             "Lab b < 该值判为蓝垫背景"),
            ("black_block_size", "局部黑边窗口(奇数)", "int", 11, 71, 2,
             "adaptiveThreshold 局部邻域大小"),
            ("black_c_offset", "局部黑边对比偏移", "float", 4.0, 35.0, 1.0,
             "↑只保留更深黑边；↓对浅灰/老化黑边更敏感"),
            ("lab_chroma_radius", "帧内白参考色度半径", "float", 8.0, 60.0, 1.0,
             "↑容忍严重偏黄老化地图；↓更严格排除彩色杂物"),
            ("morph_kernel", "形态学核大小", "int", 1, 15, 2, "去噪能力，奇数为宜"),
            ("morph_iterations", "闭运算次数", "int", 1, 5, 1, "修补白道小断裂"),
            ("yellow_min_aspect_ratio", "几何黄线最小宽高比", "float", 1.0, 10.0, 0.1,
             "排除小黄斑/黄色数字底误触发"),
            ("yellow_min_width_ratio", "几何黄线最小宽占比", "float", 0.05, 0.80, 0.02,
             "相对画面宽度，默认 0.20"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "现场黄线与 legacy_hsv 应急覆盖（仅极端暖光/眩光时应急）",
        "default_open": False,
        "items": [
            ("use_yellow_filter", "启用黄线视觉检测", "bool", 0, 1, 1, ""),
            ("yellow_h_low", "黄线 H 下限 (0-180)", "int", 0, 180, 1, "现场偏暖光可微调至 16–20"),
            ("yellow_h_high", "黄线 H 上限 (0-180)", "int", 0, 180, 1, "默认 40"),
            ("yellow_s_min", "黄线 S 下限 (0-255)", "int", 0, 255, 1, "反光泛白时调低至 55–65"),
            ("yellow_v_min", "黄线 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低至 80–95"),
            ("yellow_clear_frames", "黄线消失确认帧数", "int", 1, 60, 1, "连续多少帧看不到才解除锁存"),
            ("use_white_filter", "启用 HSV 白色过滤 (legacy_hsv)", "bool", 0, 1, 1, "仅 lane_seg_mode=legacy_hsv 时生效"),
            ("white_s_max", "白色饱和度上限 (legacy_hsv)", "int", 0, 255, 1, "仅 legacy_hsv 用：越小越严格"),
            ("white_v_min", "白色亮度下限 (legacy_hsv)", "int", 0, 255, 1, "仅 legacy_hsv 用：暗场调低"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "安全护栏（停稳判定 / 丢线保护 / 速度硬限幅）",
        "default_open": False,
        "items": [
            ("max_speed", "全局速度硬上限 (m/s)", "float", 0.1, 1.5, 0.05,
             "安全护栏：必须 ≥ 巡航速度，平时设为 0.85 即可"),
            ("min_speed", "全局速度硬下限 (m/s)", "float", 0.0, 0.5, 0.01, "通常保持 0.0"),
            ("odom_still_threshold", "停稳速度阈值 (m/s)", "float", 0.0, 0.2, 0.005,
             "低于该速度记为静止（默认 0.02）"),
            ("odom_still_hold", "停稳保持时长 (s)", "float", 0.0, 1.0, 0.05,
             "连续静止超过该时间才算真正停稳"),
            ("odom_max_age_sec", "odom 新鲜度上限 (s)", "float", 0.05, 1.0, 0.05,
             "超过该时间未收到 odom 不得判为静止"),
            ("yellow_max_hold_sec", "停车最长等待兜底 (s)", "float", 0.5, 8.0, 0.1,
             "防止极端 odom 故障导致永久卡死"),
            ("lost_line_safe_speed", "安全模式丢线爬行速 (m/s)", "float", 0.0, 0.4, 0.01,
             "丢线后的低速搜索速度"),
            ("lost_line_safe_stop_time", "安全模式丢线停机时长 (s)", "float", 0.05, 2.0, 0.05,
             "丢线超该时长立即停稳，防冲出赛道"),
        ],
    },
    {
        "tier": "calibration",
        "section": "cmd_mux_node",
        "group": "速度仲裁与安全看门狗（cmd_mux_node 双通道）",
        "default_open": False,
        "items": [
            ("stop_timeout", "停车通道超时 (s)", "float", 0.0, 2.0, 0.05,
             "estop 需超过该时长才不被恢复"),
            ("low_timeout", "巡线通道超时 (s)", "float", 0.05, 2.0, 0.05,
             "超期无新指令看门狗主动发 0，防底盘锁存"),
            ("watchdog_rate_hz", "看门狗心跳频率 (Hz)", "float", 5.0, 50.0, 1.0,
             "定期发 /cmd_mux/heartbeat 与超期零速保护"),
        ],
    },
]


def find(section, key):
    for g in SCHEMA:
        if g["section"] != section:
            continue
        for item in g["items"]:
            if item[0] == key:
                return item
    return None
