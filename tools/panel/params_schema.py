#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面板参数表单的 schema：分组、标签、范围、单位、说明。

为什么参数说明放这里而不是 YAML 注释：写回 YAML 时 PyYAML 会丢注释（DESIGN.md R08），
把说明搬到这里后，改多少次参数都不会丢。
"""

# type: float | int | bool
# (key, label, type, min, max, step, help)
SCHEMA = [
    {
        "section": "line_follower_node",
        "group": "速度与转向",
        "items": [
            ("target_speed", "目标速度", "float", 0.0, 1.0, 0.01, "直道基础速度 m/s"),
            ("max_speed", "最大速度", "float", 0.0, 1.5, 0.01, "限幅上限 m/s"),
            ("min_speed", "最小速度", "float", 0.0, 0.5, 0.01, "限幅下限 m/s"),
            ("kp", "转向比例增益", "float", 0.0, 6.0, 0.05, "yaw = -kp × 横向误差"),
            ("max_yaw_rate", "最大角速度", "float", 0.0, 3.0, 0.05, "rad/s 限幅"),
            ("speed_gain_damping", "速度随误差衰减", "float", 0.0, 1.0, 0.05,
             "误差越大越慢，0=不衰减"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "视觉与 ROI",
        "items": [
            ("roi_top_ratio", "ROI 上边界", "float", 0.0, 1.0, 0.01, "只处理画面下方这部分"),
            ("roi_bottom_ratio", "ROI 下边界", "float", 0.0, 0.5, 0.01, "底部裁掉的像素比例"),
            ("binary_threshold", "灰度阈值", "float", 0.0, 255.0, 1.0,
             "use_white_filter=false 时生效"),
            ("white_s_max", "白色饱和度上限", "int", 0, 255, 1, "越小越严格，反光易误检"),
            ("white_v_min", "白色亮度下限", "int", 0, 255, 1, "光照不足时调低"),
            ("morph_kernel", "形态学核", "int", 1, 15, 2, "去噪能力，奇数为宜"),
            ("morph_iterations", "闭运算次数", "int", 1, 5, 1, "补白线断裂"),
            ("min_contour_area", "最小轮廓面积", "float", 0.0, 20000.0, 100.0,
             "只用于 fallback 与调试图"),
            ("use_white_filter", "启用 HSV 白色过滤", "bool", 0, 1, 1,
             "建议 true，比灰度阈值抗光照"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "黄色停车线",
        "items": [
            ("use_yellow_filter", "启用黄线检测", "bool", 0, 1, 1, ""),
            ("yellow_area_ratio_threshold", "面积占比阈值", "float", 0.001, 0.30, 0.001,
             "越小越灵敏，也越容易误检"),
            ("yellow_stop_duration", "停车时长", "float", 0.0, 5.0, 0.1,
             "秒。注意：计时从“检测到”开始，不校验真实静止（R14）"),
            ("yellow_permanent_count", "永久停车线序号", "int", 1, 10, 1,
             "第几根黄线后永久停车，赛规为 4"),
            ("yellow_clear_frames", "消失确认帧数", "int", 1, 60, 1,
             "连续多少帧看不到才解除锁存，防重复计数"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "连续弯辅助",
        "items": [
            ("enable_continuous_curve_assist", "启用", "bool", 0, 1, 1, ""),
            ("reverse_curve_speed", "反向弯限速", "float", 0.0, 0.6, 0.01, "m/s"),
            ("sharp_curve_threshold", "急弯判定阈值", "float", 0.0, 1.0, 0.01, ""),
            ("sharp_curve_speed", "急弯限速", "float", 0.0, 0.6, 0.01, "m/s"),
        ],
    },
    {
        "section": "line_follower_node",
        "group": "红锥与红线",
        "items": [
            ("enable_red_filter", "启用红色检测", "bool", 0, 1, 1, ""),
            ("enable_soft_red_avoidance", "柔性避让", "bool", 0, 1, 1,
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
             "超过该时长标记 LOST（仍会前进，见 R13）"),
            ("line_lost_forward_speed", "丢线前进速度", "float", 0.0, 0.6, 0.01,
             "m/s。偏大时边界处容易冲出赛道"),
            ("line_lost_yaw_hold_factor", "丢线转向保持", "float", 0.0, 1.0, 0.05,
             "沿用上一帧转向的比例"),
        ],
    },
    {
        "section": "detection_turn_controller",
        "group": "数字转向",
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
        "group": "速度仲裁",
        "items": [
            ("stop_timeout", "停车通道超时", "float", 0.0, 2.0, 0.05,
             "秒。estop 需超过该时长才不被恢复"),
            ("high_timeout", "检测通道超时", "float", 0.0, 2.0, 0.01, "秒"),
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
