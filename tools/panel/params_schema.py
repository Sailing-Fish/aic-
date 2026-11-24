#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面板参数表单的 schema：按“正交层级 (tier)”组织，避免决赛高频参数与复赛参数混杂纠缠。

三大层级（前端渲染为独立标签页，视觉彻底隔离）：
    1. finals_core  — ★ 决赛核心调参（现场只看这一页：~19 个正交物理参数，零重叠）
    2. calibration  — ⚙ 开关与标定（赛前标定/换场地应急，标定完即冻结）
    3. legacy       — 📦 复赛归档（legacy_segment / HSV白线 / 软避障 / 独立数字转向）

类型约定：
    float | int | bool | enum
    - 数值/布尔项：(key, label, type, min, max, step, help)
    - 枚举下拉项：(key, label, "enum", options_list, None, None, help)

支持地图：
    finals_fork / finals_roundabout（决赛全功能预设）以及 fork / roundabout（复赛基线）
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
        "subtitle": "模式切换、一次标定即冻结的硬件参数、换场地抗光照/HSV 应急覆盖与看门狗",
    },
    {
        "id": "legacy",
        "label": "📦 复赛归档",
        "subtitle": "仅 planner_mode=legacy_segment / lane_seg_mode=legacy_hsv / yellow_detect_mode=legacy_area 时生效",
    },
]

# 每段一套的曲线参数（仅 legacy_segment 模式生效）。
# 键名必须与 line_follower_node.cpp load_segment_parameters() 的读取表一致（拼成 segN_<key>）。
CURVE_KEYS = [
    ("target_speed", "目标速度", "float", 0.0, 1.0, 0.01,
     "本段基础速度；超过全局 max_speed 会被限幅（启动打 WARNING）"),
    ("kp", "转向比例增益", "float", 0.0, 6.0, 0.05,
     "↑更贴弯但高速易摆；↓更稳但切弯/走外线"),
    ("steer_bias", "转向偏置 (负=向右)", "float", -1.5, 1.5, 0.05,
     "主动偏到线的一侧，选支路用；稳态偏移 = -bias/kp，"
     "太大直接切内/出界，且 |bias| 不要超过 max_yaw_rate"),
    ("max_yaw_rate", "最大角速度", "float", 0.0, 3.0, 0.05,
     "↑允许更急转向，超过舵角上限后无效；↓弯中可能转不够"),
    ("speed_gain_damping", "速度随误差衰减", "float", 0.0, 1.0, 0.05,
     "↑弯中降速更多更稳；↓保持速度更快但易切"),
    ("roi_top_ratio", "ROI 上边界", "float", 0.0, 1.0, 0.01,
     "↑看得更远、转向更早，远处噪声大；↓看得更近"),
    ("roi_bottom_ratio", "ROI 下边界", "float", 0.0, 0.5, 0.01,
     "↑近带往远处移、少看车头；↓能看到更近的黄线"),
    ("enable_continuous_curve_assist", "启用连续弯辅助", "bool", 0, 1, 1,
     "开：S 弯只跟近点；关：0.7 近 + 0.3 远"),
    ("reverse_curve_speed", "反向弯限速", "float", 0.0, 0.6, 0.01,
     "S 弯限速；↑更快但易切；↓更稳更慢"),
    ("reverse_curve_threshold", "反向弯判定阈值", "float", 0.0, 1.0, 0.01,
     "↑更晚触发反向弯辅助；↓更早、更稳"),
    ("sharp_curve_threshold", "急弯判定阈值", "float", 0.0, 1.0, 0.01,
     "↑更晚触发急弯降速；↓更早、更稳"),
    ("sharp_curve_speed", "急弯限速", "float", 0.0, 0.6, 0.01,
     "急弯限速；↑更快但易切；↓更稳更慢"),
    ("steer_ff_gain", "视觉前馈增益", "float", 0.0, 1.0, 0.01,
     "0=关闭；↑弯中更早打方向/更贴，过大易切内/摆"),
    ("steer_ff_use_v", "前馈随速度缩放", "bool", 0, 1, 1,
     "探针显示角速度闭环时开；角度映射时关"),
]

_FORK_SEG_TITLES = {
    1: "段1 · 起步直道（0→2.33 m，legacy 分段）",
    2: "段2 · 选支/转弯（legacy 分段）",
    3: "段3 · 出分支后（legacy 分段）",
}

_ROUNDABOUT_SEG_TITLES = {
    1: "段1 · 第2根黄线前（legacy 分段）",
    2: "段2 · 黄线2→3（legacy 分段）",
    3: "段3 · 黄线3→环岛二出口（legacy 分段）",
    4: "段4 · 环岛二出口→终点（legacy 分段）",
}

SEGMENT_TITLES = {
    "fork": _FORK_SEG_TITLES,
    "finals_fork": _FORK_SEG_TITLES,
    "roundabout": _ROUNDABOUT_SEG_TITLES,
    "finals_roundabout": _ROUNDABOUT_SEG_TITLES,
}

_FORK_BOUNDARIES = [
    ("seg2_start_count", "段2 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段2；岔路图默认 2"),
    ("seg3_start_dist_m", "段3 起始里程 (m)", "float", 0.1, 6.0, 0.1,
     "第2根黄线之后再走多少米进入段3；0 = 改看 seg3_start_count"),
    ("seg3_start_count", "段3 起始黄线序号", "int", 0, 4, 1,
     "0 = 不用黄线判定（岔路走里程）"),
]

_ROUNDABOUT_BOUNDARIES = [
    ("seg2_start_count", "段2 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段2；默认 2"),
    ("seg3_start_count", "段3 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段3；默认 3"),
    ("seg4_start_dist_m", "段4 起始里程 (m)", "float", 1.0, 6.0, 0.1,
     "距第3根黄线再走多少米进入段4（环岛二出口）"),
]

SEGMENT_BOUNDARIES = {
    "fork": _FORK_BOUNDARIES,
    "finals_fork": _FORK_BOUNDARIES,
    "roundabout": _ROUNDABOUT_BOUNDARIES,
    "finals_roundabout": _ROUNDABOUT_BOUNDARIES,
}

_FORK_SEG2_EXTRA = [
    ("steer_bias_dist_m", "偏置1 持续里程 (m)", "float", 0.0, 2.0, 0.05,
     "进入本段后施加偏置1 走多少米；0 = 整段施加"),
    ("steer_bias_gap_m", "两段偏置的间隔 (m)", "float", 0.0, 2.0, 0.05,
     "偏置1 打完先松多少米，再打偏置2"),
    ("steer_bias2", "转向偏置2 (负=向右)", "float", -1.5, 1.5, 0.05,
     "与偏置1 同号=再拐一次，反号=拐回来"),
    ("steer_bias2_dist_m", "偏置2 持续里程 (m)", "float", 0.0, 2.0, 0.05,
     "0 = 与偏置1 相同"),
]

SEGMENT_EXTRA = {
    ("fork", 2): _FORK_SEG2_EXTRA,
    ("finals_fork", 2): _FORK_SEG2_EXTRA,
}


def _legacy_segment_groups():
    groups = []
    for map_name, titles in SEGMENT_TITLES.items():
        for seg, title in titles.items():
            items = [("seg%d_%s" % (seg, k[0]),) + k[1:] for k in CURVE_KEYS]
            items += [("seg%d_%s" % (seg, k[0]),) + k[1:]
                      for k in SEGMENT_EXTRA.get((map_name, seg), [])]
            groups.append({
                "tier": "legacy",
                "section": "line_follower_node",
                "group": title,
                "segment": seg,
                "maps": [map_name],
                "items": items,
            })
    for map_name, items in SEGMENT_BOUNDARIES.items():
        groups.append({
            "tier": "legacy",
            "section": "line_follower_node",
            "group": "分段切换条件（legacy_segment）",
            "maps": [map_name],
            "items": items,
        })
    return groups


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
    # TIER 2: ⚙ 模式开关、硬件标定与换场地应急（平时无需改动）
    # =====================================================================
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "模式总开关（一键切换 / 回退复赛）",
        "default_open": True,
        "items": [
            ("planner_mode", "规划器模式", "enum",
             ["finals_corridor", "legacy_segment"], None, None,
             "finals_corridor=决赛动态走廊+FinishGuard；legacy_segment=一键回退复赛分段"),
            ("lane_seg_mode", "车道视觉分割模式", "enum",
             ["relative_border", "hybrid", "legacy_hsv"], None, None,
             "relative_border 抗光照/泛黄/眩光（零手动标定）；hybrid 结合宽松 HSV；legacy_hsv 旧版"),
            ("yellow_detect_mode", "黄线检测模式", "enum",
             ["geometric_cv", "yolo_cv_hybrid", "legacy_area"], None, None,
             "geometric_cv 横向几何+底边精准测距；11类 YOLO 就绪后切 yolo_cv_hybrid"),
            ("enable_semantic_hole_fill", "启用语义回填(防数字挖洞)", "bool", 0, 1, 1,
             "把检测到的彩色/涂鸦数字区域回填白色，防止赛道二值图断裂"),
            ("enable_obstacle_memory", "启用障碍物世界坐标记忆", "bool", 0, 1, 1,
             "在 /odom_combined 保留车侧后轮区间障碍，防后轮切内蹭 +30s 禁行区"),
            ("enable_cv_red_fallback", "启用白道 CV 红障兜底 (YOLO||CV)", "bool", 0, 1, 1,
             "开：YOLO 或白道内 CV 任一检到红障即避让（YOLO优先、CV补漏，并输出 [DEC] red_src 来源日志）"),
            ("allow_dual_digit_cover", "允许直道双数字同压", "bool", 0, 1, 1,
             "关=直道 10cm 双数字时偏向大数字避开小数字；开=允许同时覆盖"),
            ("use_homography", "启用 4 点 IPM 单应矩阵", "bool", 0, 1, 1,
             "true 使用 calibrate_homography.py 标定的矩阵；false 用针孔模型回退"),
            ("yellow_require_odom_still", "黄线严格要求里程计停稳", "bool", 0, 1, 1,
             "决赛推荐 true：未停稳前不提前放行，杜绝 +15s 未停稳罚时"),
            ("lost_line_safe_mode", "丢线安全减速模式", "bool", 0, 1, 1,
             "决赛推荐 true：丢线迅速减速并短时停机，防冲出赛道触发 +15s/s"),
        ],
    },
    {
        "tier": "calibration",
        "section": "line_follower_node",
        "group": "底盘与刹车物理标定（实测一次即冻结）",
        "desc": "由 speed_step.py / brake_probe.py 一次测准后不要在赛场反复猜测",
        "default_open": False,
        "items": [
            ("odom_scale_factor", "里程计卷尺标定比例", "float", 0.70, 1.30, 0.005,
             "实测卷尺距离 / 里程计读数；修正终点门控与障碍记忆漂移"),
            ("yellow_brake_decel_mps2", "实测刹车减速度 (m/s²)", "float", 0.3, 5.0, 0.1,
             "用于物理公式 v²/(2a) 计算刹车触发距离；由 brake_probe.py 测得"),
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
        "group": "抗光照视觉微调（relative_border / geometric_cv）",
        "desc": "仅在换新场地出现明显白道漏检/蓝垫误检时微调",
        "default_open": False,
        "items": [
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
        "group": "现场色彩 HSV 应急覆盖（仅极端暖光/眩光时应急）",
        "default_open": False,
        "items": [
            ("use_yellow_filter", "启用黄线视觉检测", "bool", 0, 1, 1, ""),
            ("yellow_h_low", "黄线 H 下限 (0-180)", "int", 0, 180, 1, "现场偏暖光可微调至 16–20"),
            ("yellow_h_high", "黄线 H 上限 (0-180)", "int", 0, 180, 1, "默认 40"),
            ("yellow_s_min", "黄线 S 下限 (0-255)", "int", 0, 255, 1, "反光泛白时调低至 55–65"),
            ("yellow_v_min", "黄线 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低至 80–95"),
            ("yellow_clear_frames", "黄线消失确认帧数", "int", 1, 60, 1, "连续多少帧看不到才解除锁存"),
            ("enable_red_filter", "启用红色 HSV 兜底", "bool", 0, 1, 1, "11类 YOLO 就绪前应急用"),
            ("red_h_low_1", "红色段1 H 下限", "int", 0, 30, 1, "默认 0"),
            ("red_h_high_1", "红色段1 H 上限", "int", 0, 30, 1, "默认 10"),
            ("red_h_low_2", "红色段2 H 下限", "int", 140, 180, 1, "默认 160"),
            ("red_h_high_2", "红色段2 H 上限", "int", 140, 180, 1, "默认 180"),
            ("red_s_min", "红色 S 下限 (0-255)", "int", 0, 255, 1, "反光发白可调低至 70"),
            ("red_v_min", "红色 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低至 50"),
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
        "group": "速度仲裁与安全看门狗（cmd_mux_node）",
        "default_open": False,
        "items": [
            ("enable_detection_override", "允许检测通道抢占", "bool", 0, 1, 1,
             "决赛走廊模式必须 false（防止数字转向压制红障避让）；复赛 legacy 为 true"),
            ("stop_timeout", "停车通道超时 (s)", "float", 0.0, 2.0, 0.05,
             "estop 需超过该时长才不被恢复"),
            ("high_timeout", "检测通道超时 (s)", "float", 0.0, 2.0, 0.01, "legacy 抢占通道超时"),
            ("low_timeout", "巡线通道超时 (s)", "float", 0.05, 2.0, 0.05,
             "超期无新指令看门狗主动发 0，防底盘锁存"),
            ("watchdog_rate_hz", "看门狗心跳频率 (Hz)", "float", 5.0, 50.0, 1.0,
             "定期发 /cmd_mux/heartbeat 与超期零速保护"),
        ],
    },

    # =====================================================================
    # TIER 3: 📦 复赛 / Legacy 归档（决赛模式下完全不生效，独立收起）
    # =====================================================================
] + _legacy_segment_groups() + [
    {
        "tier": "legacy",
        "section": "line_follower_node",
        "group": "复赛黄线计数与面积阈值（legacy_area / legacy_segment）",
        "default_open": False,
        "items": [
            ("yellow_area_ratio_threshold", "面积占比阈值 (legacy)", "float", 0.001, 0.30, 0.001,
             "仅 yellow_detect_mode=legacy_area 生效；决赛请改调「车头留白距离」"),
            ("yellow_permanent_count", "永久停车线序号 (legacy)", "int", 1, 10, 1,
             "仅 legacy_segment 生效；决赛改由 FinishGuard 里程+封口判定"),
            ("yellow_stop_duration", "停车兜底时长 (legacy)", "float", 0.0, 5.0, 0.1,
             "legacy 模式 odom 丢失时兜底等待"),
        ],
    },
    {
        "tier": "legacy",
        "section": "line_follower_node",
        "group": "复赛 HSV 白线阈值（legacy_hsv）",
        "default_open": False,
        "items": [
            ("use_white_filter", "启用 HSV 白色过滤", "bool", 0, 1, 1, "legacy_hsv 模式下建议 true"),
            ("white_s_max", "白色饱和度上限", "int", 0, 255, 1, "legacy_hsv 用：越小越严格"),
            ("white_v_min", "白色亮度下限", "int", 0, 255, 1, "legacy_hsv 用：暗场调低"),
            ("binary_threshold", "灰度阈值", "float", 0.0, 255.0, 1.0, "use_white_filter=false 时生效"),
            ("min_contour_area", "最小轮廓面积", "float", 0.0, 20000.0, 100.0, "只用于 fallback 与调试图"),
        ],
    },
    {
        "tier": "legacy",
        "section": "line_follower_node",
        "group": "复赛柔性红障避让（legacy_segment）",
        "default_open": False,
        "items": [
            ("enable_soft_red_avoidance", "柔性避让 (legacy)", "bool", 0, 1, 1,
             "仅 legacy_segment 生效；决赛请用走廊避障余量与限速"),
            ("red_near_ratio", "近场判定", "float", 0.0, 1.0, 0.01, "bbox 底部到达该比例开始避让"),
            ("red_very_close_ratio", "贴脸判定", "float", 0.0, 1.0, 0.01, "超过该比例强制避让"),
            ("red_soft_yaw", "基础避让角速度", "float", 0.0, 1.0, 0.01, "rad/s"),
            ("red_repulsion_gain", "斥力放大", "float", 0.0, 5.0, 0.1, "越近越强"),
            ("red_corridor_half_width_ratio", "走廊半宽", "float", 0.0, 1.0, 0.01, "只避让挡在走廊里的目标"),
            ("red_soft_speed", "避让速度", "float", 0.0, 0.6, 0.01, "m/s"),
            ("red_min_forward_speed", "最低前进速度", "float", 0.01, 0.5, 0.01, "红目标只减速不停车"),
            ("red_lane_keep_min_factor", "巡线力保留比例", "float", 0.0, 1.0, 0.05, "避让时保留巡线转向比例"),
            ("red_cone_area_ratio_threshold", "红锥面积占比", "float", 0.0, 0.01, 0.0001, "单个轮廓 / ROI"),
            ("red_line_aspect_ratio_min", "红线宽高比", "float", 1.0, 10.0, 0.1, "横向长条判据"),
            ("red_line_min_width_ratio", "红线最小宽度占比", "float", 0.0, 1.0, 0.01, "相对画面宽度"),
        ],
    },
    {
        "tier": "legacy",
        "section": "line_follower_node",
        "group": "复赛丢线与独立数字转向（legacy）",
        "default_open": False,
        "items": [
            ("line_lost_stop_time", "丢线判定时长 (legacy)", "float", 0.0, 3.0, 0.05, "超过该时长标记 LOST"),
            ("line_lost_forward_speed", "丢线前进速度 (legacy)", "float", 0.0, 0.6, 0.01, "m/s"),
            ("line_lost_yaw_hold_factor", "丢线转向保持", "float", 0.0, 1.0, 0.05, "沿用上一帧转向比例"),
        ],
    },
    {
        "tier": "legacy",
        "section": "detection_turn_controller",
        "group": "独立数字转向控制器（legacy detection_turn_controller）",
        "default_open": False,
        "items": [
            ("confidence_threshold", "置信度阈值", "float", 0.0, 1.0, 0.01, ""),
            ("frame_window", "累积帧数", "int", 1, 20, 1, "多少帧后触发转向"),
            ("turn_duration", "转向时长 (s)", "float", 0.0, 2.0, 0.01, "开环定时，随速度变化敏感"),
            ("turn_linear", "转向时线速度 (m/s)", "float", 0.0, 0.6, 0.01, ""),
            ("yaw_gain", "转向增益", "float", 0.0, 5.0, 0.1, ""),
            ("max_yaw_rate", "最大角速度 (rad/s)", "float", 0.0, 3.0, 0.05, ""),
            ("turn_cooldown", "冷却时间 (s)", "float", 0.0, 3.0, 0.05, ""),
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
