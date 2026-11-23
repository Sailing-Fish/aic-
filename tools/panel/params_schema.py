#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面板参数表单的 schema：分组、标签、范围、单位、说明。

为什么参数说明放这里而不是 YAML 注释：写回 YAML 时 PyYAML 会丢注释（DESIGN.md R08），
把说明搬到这里后，改多少次参数都不会丢。

类型约定：
    float | int | bool | enum
    - 数值/布尔项：(key, label, type, min, max, step, help)
    - 枚举下拉项：(key, label, "enum", options_list, None, None, help)

支持地图：
    fork / roundabout（复赛基线）以及 finals_fork / finals_roundabout（决赛全功能预设）
"""

# 每段一套的曲线参数。键名必须与 line_follower_node.cpp
# load_segment_parameters() 的读取表一致（那边拼成 segN_<key>）。
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

# 每张地图的段标题。段号 -> 标题（同时决定这张图有几段）
_FORK_SEG_TITLES = {
    1: "段1 · 起步直道（0→2.33 m，可提速）",
    2: "段2 · 选支/转弯（第2根黄线后 seg3_start_dist_m 米内）",
    3: "段3 · 出分支后（恢复正常跟线）",
}

_ROUNDABOUT_SEG_TITLES = {
    1: "段1 · 第2根黄线前（直道+环岛一）",
    2: "段2 · 黄线2→3（U弯+环岛二入口）",
    3: "段3 · 黄线3→环岛二出口",
    4: "段4 · 环岛二出口→终点",
}

SEGMENT_TITLES = {
    "fork": _FORK_SEG_TITLES,
    "finals_fork": _FORK_SEG_TITLES,
    "roundabout": _ROUNDABOUT_SEG_TITLES,
    "finals_roundabout": _ROUNDABOUT_SEG_TITLES,
}

_FORK_BOUNDARIES = [
    ("seg2_start_count", "段2 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段2；岔路图默认 2（第1根在起步 0.77 m 处）"),
    ("seg3_start_dist_m", "段3 起始里程 (m)", "float", 0.1, 6.0, 0.1,
     "第2根黄线之后再走多少米进入段3（选支/转弯结束）；"
     "填 0 等于关掉，改看 seg3_start_count"),
    ("seg3_start_count", "段3 起始黄线序号", "int", 0, 4, 1,
     "0 = 不用黄线判定（岔路走里程）；环岛图不显示这一项"),
]

_ROUNDABOUT_BOUNDARIES = [
    ("seg2_start_count", "段2 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段2；默认 2"),
    ("seg3_start_count", "段3 起始黄线序号", "int", 1, 4, 1,
     "跨过第 N 根黄线进入段3；默认 3"),
    ("seg4_start_dist_m", "段4 起始里程 (m)", "float", 1.0, 6.0, 0.1,
     "距第3根黄线再走多少米进入段4（环岛二出口）；"
     "= 2.27×1.08 + 检测提前量 + 0.15，宁晚不早"),
]

# 分段切换条件（只对该地图显示）
SEGMENT_BOUNDARIES = {
    "fork": _FORK_BOUNDARIES,
    "finals_fork": _FORK_BOUNDARIES,
    "roundabout": _ROUNDABOUT_BOUNDARIES,
    "finals_roundabout": _ROUNDABOUT_BOUNDARIES,
}

# 只给某一张图的某一段追加的参数（(地图, 段号) -> items）
_FORK_SEG2_EXTRA = [
    ("steer_bias_dist_m", "偏置1 持续里程 (m)", "float", 0.0, 2.0, 0.05,
     "进入本段后施加偏置1 走多少米；0 = 整段都施加（等于没有时序）"),
    ("steer_bias_gap_m", "两段偏置的间隔 (m)", "float", 0.0, 2.0, 0.05,
     "偏置1 打完先松多少米（顺着线走），再打偏置2"),
    ("steer_bias2", "转向偏置2 (负=向右)", "float", -1.5, 1.5, 0.05,
     "拐出来用；与偏置1 同号=再拐一次，反号=拐回来"),
    ("steer_bias2_dist_m", "偏置2 持续里程 (m)", "float", 0.0, 2.0, 0.05,
     "0 = 与偏置1 相同（对称脉冲）"),
]

SEGMENT_EXTRA = {
    ("fork", 2): _FORK_SEG2_EXTRA,
    ("finals_fork", 2): _FORK_SEG2_EXTRA,
}


def _segment_groups():
    groups = []
    for map_name, titles in SEGMENT_TITLES.items():
        for seg, title in titles.items():
            items = [("seg%d_%s" % (seg, k[0]),) + k[1:] for k in CURVE_KEYS]
            items += [("seg%d_%s" % (seg, k[0]),) + k[1:]
                      for k in SEGMENT_EXTRA.get((map_name, seg), [])]
            groups.append({
                "section": "line_follower_node",
                "group": title,
                "segment": seg,
                "maps": [map_name],
                "items": items,
            })
    for map_name, items in SEGMENT_BOUNDARIES.items():
        groups.append({
            "section": "line_follower_node",
            "group": "分段切换（legacy_segment 模式生效）",
            "maps": [map_name],
            "items": items,
        })
    return groups


SCHEMA = [
    {
        "section": "line_follower_node",
        "group": "决赛模式总开关（一键切换 / 回退）",
        "items": [
            ("planner_mode", "规划器模式", "enum",
             ["legacy_segment", "finals_corridor"], None, None,
             "legacy_segment=复赛分段回退；finals_corridor=决赛动态走廊+障碍记忆+FinishGuard"),
            ("lane_seg_mode", "车道视觉分割模式", "enum",
             ["legacy_hsv", "relative_border", "hybrid"], None, None,
             "relative_border 抗光照/老化/眩光（零手动标定）；hybrid 结合宽松 HSV；legacy_hsv 旧版回退"),
            ("yellow_detect_mode", "黄线检测模式", "enum",
             ["legacy_area", "geometric_cv", "yolo_cv_hybrid"], None, None,
             "geometric_cv 横向细长几何+底边精准测距；11类YOLO就绪后切 yolo_cv_hybrid"),
            ("enable_semantic_hole_fill", "启用语义回填(防数字挖洞)", "bool", 0, 1, 1,
             "把检测到的彩色/涂鸦数字卡片区域回填白色，防赛道二值图断裂"),
            ("enable_obstacle_memory", "启用障碍物世界坐标记忆", "bool", 0, 1, 1,
             "在 /odom_combined 保留车侧后轮区间 [-0.28m,0] 障碍，防切内蹭+30s禁行区"),
            ("allow_dual_digit_cover", "允许直道双数字同压", "bool", 0, 1, 1,
             "关=直道10cm双数字时偏向大数字避开小数字；开=按规则允许同时覆盖"),
            ("use_homography", "启用 4 点 IPM 单应矩阵", "bool", 0, 1, 1,
             "true 使用 calibrate_homography.py 标定的 homography_matrix；false 自动用针孔回退"),
            ("yellow_require_odom_still", "黄线严格要求里程计停稳", "bool", 0, 1, 1,
             "决赛推荐 true：未停稳前不因超时提前放行，杜绝 +15s 未停稳罚时"),
            ("lost_line_safe_mode", "丢线安全减速模式", "bool", 0, 1, 1,
             "决赛推荐 true：丢线时迅速降速并在短超时后停车，防冲出赛道触发 +15s/s"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "决赛走廊规划与 FinishGuard",
        "items": [
            ("target_speed", "走廊巡航速度 (m/s)", "float", 0.1, 1.0, 0.01,
             "finals_corridor 模式基础巡航速度（Run 1 保底建议 0.40-0.44）"),
            ("obstacle_speed_mps", "红锥绕行限速 (m/s)", "float", 0.05, 0.6, 0.01,
             "走廊内出现红锥时的速度上限"),
            ("no_entry_speed_mps", "禁行区绕行限速 (m/s)", "float", 0.05, 0.5, 0.01,
             "走廊内出现 +30s 红禁行区时的速度上限"),
            ("digit_approach_speed_mps", "吃数字接近限速 (m/s)", "float", 0.05, 0.6, 0.01,
             "直道对准高分 Bonus 数字时的限速"),
            ("border_margin_m", "车道边线安全余量 (m)", "float", 0.0, 0.08, 0.005,
             "车身外缘到黑线的保护余量，默认 0.025m"),
            ("cone_margin_m", "红锥侧向余量 (m)", "float", 0.0, 0.08, 0.005,
             "绕行红锥时的额外侧向保护余量"),
            ("no_entry_margin_m", "禁行区侧向余量 (m)", "float", 0.0, 0.08, 0.005,
             "绕行 +30s 空心红禁行区的额外侧向余量"),
            ("corridor_kp_y", "走廊横向纠偏增益", "float", 0.5, 8.0, 0.1,
             "走廊模式下目标横向偏移 y_ref 的转向比例增益"),
            ("corridor_heading_gain", "走廊航向阻尼增益", "float", 0.0, 3.0, 0.05,
             "抑制过冲与摆头"),
            ("digit_min_conf", "Bonus 数字最低置信度", "float", 0.1, 0.95, 0.05,
             "低于该置信度不触发吃数字偏移"),
            ("max_digit_lateral_shift_m", "吃数字最大横向偏移 (m)", "float", 0.0, 0.20, 0.01,
             "限制为吃数字偏离中线的最大幅度，避免压线"),
            ("odom_scale_factor", "里程计卷尺标定比例", "float", 0.7, 1.3, 0.005,
             "由 speed_step.py --ask-tape 实测得到，修正里程累积"),
            ("finish_min_dist_m", "终点门控最短里程 (m)", "float", 4.0, 15.0, 0.1,
             "累计里程达该值后才允许视觉封口判终点（防止随机直道黄线误停赛）"),
            ("finish_force_dist_m", "终点强判兜底里程 (m)", "float", 6.0, 18.0, 0.1,
             "超过该里程遇到黄线强制判终点"),
            ("yellow_pre_brake_dist_m", "黄线提前减速距离 (m)", "float", 0.2, 1.5, 0.05,
             "车头距黄线小于该距离进入爬行限速段"),
            ("yellow_creep_speed_mps", "黄线前爬行速度 (m/s)", "float", 0.08, 0.45, 0.01,
             "预减速段速度，确保零压线稳停"),
            ("yellow_bumper_margin_m", "车头防撞条余量 (m)", "float", 0.02, 0.25, 0.01,
             "停在黄线前的安全留白，防止车头压线触发 +15s"),
        ],
    },
] + _segment_groups() + [
    {
        "section": "line_follower_node",
        "group": "黄线停车",
        "items": [
            ("yellow_area_ratio_threshold", "面积占比阈值", "float", 0.001, 0.30, 0.001,
             "↓更早检测黄线/更早停；↑更晚、易压线"),
            ("yellow_stop_min_time", "最短停车时间", "float", 0.0, 2.0, 0.05,
             "↑更稳更慢；不能低于 0.30 s"),
            ("odom_still_threshold", "静止速度阈值", "float", 0.0, 0.2, 0.005,
             "↑更容易判静止；↓要求更严格"),
            ("odom_still_hold", "静止保持时间", "float", 0.0, 1.0, 0.05,
             "↑要静止更久、更稳；↓放行更快"),
            ("odom_max_age_sec", "odom 新鲜度上限 (s)", "float", 0.05, 1.0, 0.05,
             "超过该时间未收到 odom 不得判为静止"),
            ("yellow_stop_duration", "停车兜底时长", "float", 0.0, 5.0, 0.1,
             "legacy 模式下 odom 丢失时兜底等待；↑更稳、↓更快"),
            ("yellow_max_hold_sec", "决赛最长等待兜底 (s)", "float", 0.5, 8.0, 0.1,
             "finals 模式下防止极端异常永久卡死的最长保持时间"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "黄线刹车（反拖，替代直接发 0）",
        "items": [
            ("yellow_brake_speed", "反拖速度 (m/s)", "float", 0.0, 1.5, 0.05,
             "检测到黄线后发的负速度大小；0 = 关闭（退回直接发 0）。"
             "先跑 tools/brake_probe.py --probe-only 确认固件接受反转"),
            ("yellow_brake_period_s", "发送周期 (s)", "float", 0.0, 1.0, 0.02,
             "反拖指令多久重发一次；0 = 每帧都发"),
            ("yellow_brake_pulse_s", "每周期反拖时长 (s)", "float", 0.0, 1.0, 0.02,
             "每个周期里发多久负速度，其余时间发 0（点刹）；0 或 ≥周期 = 连续反拖"),
            ("yellow_brake_release_speed", "撤回阈值 (m/s)", "float", 0.0, 0.4, 0.01,
             "odom 速度降到它以下立刻停止反拖（防止顶成倒车）；0 = 不用这个条件"),
            ("yellow_brake_max_time_s", "最长反拖时间 (s)", "float", 0.0, 3.0, 0.05,
             "兜底：最多反拖这么久，到点无条件撤回"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "速度限幅（全段共用）",
        "items": [
            ("max_speed", "最大速度", "float", 0.0, 1.5, 0.01,
             "限幅上限 m/s，必须 ≥ 各段目标速度"),
            ("min_speed", "最小速度", "float", 0.0, 0.5, 0.01, "限幅下限 m/s"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "视觉阈值与相对分割微调",
        "items": [
            ("binary_threshold", "灰度阈值", "float", 0.0, 255.0, 1.0,
             "use_white_filter=false 时生效"),
            ("white_s_max", "白色饱和度上限", "int", 0, 255, 1, "legacy_hsv 用：越小越严格，反光易误检"),
            ("white_v_min", "白色亮度下限", "int", 0, 255, 1, "legacy_hsv 用：光照不足时调低"),
            ("blue_margin", "蓝垫色差抑制阈值", "int", 5, 80, 1, "relative_border 用：B-max(R,G) > 该值判蓝垫"),
            ("lab_blue_b_max", "Lab b 通道蓝垫上限", "int", 80, 130, 1, "relative_border 用：b < 该值判蓝垫"),
            ("black_block_size", "局部黑边窗口(奇数)", "int", 11, 71, 2, "adaptiveThreshold 局部邻域大小"),
            ("black_c_offset", "局部黑边对比偏移", "float", 4.0, 35.0, 1.0, "↑只保留更深黑边；↓对浅灰边更敏感"),
            ("lab_chroma_radius", "帧内白参考色度半径", "float", 8.0, 60.0, 1.0, "↑容忍偏黄老化；↓排除彩色杂物"),
            ("morph_kernel", "形态学核", "int", 1, 15, 2, "去噪能力，奇数为宜"),
            ("morph_iterations", "闭运算次数", "int", 1, 5, 1, "补白线断裂"),
            ("min_contour_area", "最小轮廓面积", "float", 0.0, 20000.0, 100.0,
             "只用于 fallback 与调试图"),
            ("use_white_filter", "启用 HSV 白色过滤", "bool", 0, 1, 1,
             "legacy_hsv 模式下建议 true，比灰度阈值抗光照"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "黄色停车线（含现场 HSV 应急覆盖）",
        "items": [
            ("use_yellow_filter", "启用黄线检测", "bool", 0, 1, 1, ""),
            ("yellow_h_low", "黄线 H 下限 (0-180)", "int", 0, 180, 1, "现场偏暖光可微调至 16-20"),
            ("yellow_h_high", "黄线 H 上限 (0-180)", "int", 0, 180, 1, "默认 40"),
            ("yellow_s_min", "黄线 S 下限 (0-255)", "int", 0, 255, 1, "反光泛白时适当调低（如 60）"),
            ("yellow_v_min", "黄线 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低（如 90）"),
            ("yellow_min_aspect_ratio", "几何黄线最小宽高比", "float", 1.0, 10.0, 0.1,
             "geometric_cv 用：排除小黄斑/黄色数字底误触发"),
            ("yellow_min_width_ratio", "几何黄线最小宽占比", "float", 0.05, 0.8, 0.02,
             "相对画面宽度，默认 0.20"),
            ("yellow_permanent_count", "永久停车线序号 (legacy)", "int", 1, 10, 1,
             "仅 legacy_segment 模式生效；finals 模式改由 FinishGuard 判定"),
            ("yellow_clear_frames", "消失确认帧数", "int", 1, 60, 1,
             "连续多少帧看不到才解除锁存，防重复计数"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "红锥与红线（含现场 HSV 应急覆盖）",
        "items": [
            ("enable_red_filter", "启用红色检测", "bool", 0, 1, 1, ""),
            ("red_h_low_1", "红色段1 H 下限", "int", 0, 30, 1, "默认 0"),
            ("red_h_high_1", "红色段1 H 上限", "int", 0, 30, 1, "默认 10"),
            ("red_h_low_2", "红色段2 H 下限", "int", 140, 180, 1, "默认 160"),
            ("red_h_high_2", "红色段2 H 上限", "int", 140, 180, 1, "默认 180"),
            ("red_s_min", "红色 S 下限 (0-255)", "int", 0, 255, 1, "反光发白可调低至 70"),
            ("red_v_min", "红色 V 下限 (0-255)", "int", 0, 255, 1, "暗场可调低至 50"),
            ("enable_soft_red_avoidance", "柔性避让 (legacy)", "bool", 0, 1, 1,
             "关掉则用旧的固定时长转向"),
            ("red_near_ratio", "近场判定", "float", 0.0, 1.0, 0.01,
             "bbox 底部到达该比例开始避让"),
            ("red_very_close_ratio", "贴脸判定", "float", 0.0, 1.0, 0.01,
             "超过该比例忽略走廊判断强制避让"),
            ("red_soft_yaw", "基础避让角速度", "float", 0.0, 1.0, 0.01, "rad/s"),
            ("red_repulsion_gain", "斥力放大", "float", 0.0, 5.0, 0.1, "越近越强"),
            ("red_corridor_half_width_ratio", "走廊半宽", "float", 0.0, 1.0, 0.01,
             "只避让挡在行驶走廊里的目标"),
            ("red_soft_speed", "避让速度", "float", 0.0, 0.6, 0.01, "m/s"),
            ("red_min_forward_speed", "最低前进速度", "float", 0.01, 0.5, 0.01,
             "红目标只减速不停车"),
            ("red_lane_keep_min_factor", "巡线力保留比例", "float", 0.0, 1.0, 0.05,
             "避让时至少保留多少巡线转向"),
            ("red_cone_area_ratio_threshold", "红锥面积占比", "float", 0.0, 0.01, 0.0001,
             "单个轮廓 / ROI"),
            ("red_line_aspect_ratio_min", "红线宽高比", "float", 1.0, 10.0, 0.1,
             "横向长条判据"),
            ("red_line_min_width_ratio", "红线最小宽度占比", "float", 0.0, 1.0, 0.01,
             "相对画面宽度"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "丢线处理",
        "items": [
            ("line_lost_stop_time", "丢线判定时长", "float", 0.0, 3.0, 0.05,
             "超过该时长标记 LOST"),
            ("line_lost_forward_speed", "丢线前进速度", "float", 0.0, 0.6, 0.01,
             "m/s。偏大时边界处容易冲出赛道"),
            ("line_lost_yaw_hold_factor", "丢线转向保持", "float", 0.0, 1.0, 0.05,
             "沿用上一帧转向的比例"),
            ("lost_line_safe_speed", "安全模式丢线爬行速", "float", 0.0, 0.4, 0.01,
             "lost_line_safe_mode=true 时的低速搜索速度"),
            ("lost_line_safe_stop_time", "安全模式丢线停机时长", "float", 0.05, 2.0, 0.05,
             "丢线超过该时长直接停稳，防冲出赛道"),
        ],
    },
    {
        "section": "detection_turn_controller",
        "group": "数字转向（legacy 独立控制器）",
        "items": [
            ("confidence_threshold", "置信度阈值", "float", 0.0, 1.0, 0.01, ""),
            ("frame_window", "累积帧数", "int", 1, 20, 1, "多少帧后触发转向"),
            ("turn_duration", "转向时长", "float", 0.0, 2.0, 0.01,
             "秒。开环定时，随速度变化敏感"),
            ("turn_linear", "转向时线速度", "float", 0.0, 0.6, 0.01, "m/s"),
            ("yaw_gain", "转向增益", "float", 0.0, 5.0, 0.1, ""),
            ("max_yaw_rate", "最大角速度", "float", 0.0, 3.0, 0.05, "rad/s"),
            ("turn_cooldown", "冷却时间", "float", 0.0, 3.0, 0.05, "秒"),
        ],
    },
    {
        "section": "cmd_mux_node",
        "group": "速度仲裁与安全看门狗",
        "items": [
            ("enable_detection_override", "允许检测通道抢占", "bool", 0, 1, 1,
             "决赛走廊模式必须 false（避免数字转向压制红障避让）；复赛 legacy 为 true"),
            ("stop_timeout", "停车通道超时", "float", 0.0, 2.0, 0.05,
             "秒。estop 需超过该时长才不被恢复"),
            ("high_timeout", "检测通道超时", "float", 0.0, 2.0, 0.01, "秒"),
            ("low_timeout", "巡线通道超时", "float", 0.05, 2.0, 0.05,
             "超过该时长无新指令，看门狗主动发 0 防底盘锁存"),
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
