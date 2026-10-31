#pragma once

#include <algorithm>

namespace line_follower
{

// ============================================================
// 分段参数
//
// 一张地图被切成若干段，每段一套"对弯道影响最大"的参数。
// 字段与 YAML 的 segN_<key> 一一对应，也与面板"弯道调试"组一一对应；
// 三处（本文件 / line_follower_node.cpp 的加载表 / params_schema.py）必须同名。
// ============================================================
struct SegmentParams
{
  double target_speed{0.50};
  double kp{3.00};
  // 主动转向偏置 rad/s，与 angular.z 同号：负 = 向右，正 = 向左。
  // 稳态平衡点 error = -steer_bias / kp，用来"故意偏到线的一侧"（选支路）。
  // steer_bias2 给"拐进去再拐出来"的第二种偏置用，配合两个持续里程。
  double steer_bias{0.0};
  double steer_bias2{0.0};
  double bias_phase1_dist_m{0.0};
  double bias_gap_m{0.0};
  double bias_phase2_dist_m{0.0};
  double max_yaw_rate{1.50};
  double speed_gain_damping{0.50};
  double roi_top_ratio{0.65};
  double roi_bottom_ratio{0.20};
  double reverse_curve_speed{0.34};
  double reverse_curve_threshold{0.35};
  double sharp_curve_threshold{0.50};
  double sharp_curve_speed{0.36};
  double steer_ff_gain{0.0};
  bool continuous_curve_assist{true};
  bool steer_ff_use_v{false};
};

constexpr int kMaxSegments = 4;

// ============================================================
// 段内"转向偏置时序"：同一段里打两个短脉冲，中间留一个间隔。
//
//   0 .. D1              用 B1（拐进支路）
//   D1 .. D1+G           不施加（间隔：靠 P 跟线通过这一小段）
//   D1+G .. D1+G+D2      用 B2（B2 与 B1 反号 = 拐出来；同号 = 再拐一次）
//   之后                 0
//
// D1 <= 0            ：整段都用 B1（不切换）
// G  < 0             ：按 0 处理
// D2 <= 0            ：与 D1 相同（对称脉冲）
//
// 注意：这是"持续施加的角速度偏置"，不是脉冲；实际转角 ≈ B × (D / v)，
// 窗口短于 ~0.2 s 会被舵机一阶滞后吃掉，所以 D 一般 0.2–0.4 m。
// ============================================================
inline double steer_bias_at(
  double seg_dist_m, double b1, double d1, double gap, double b2, double d2)
{
  if (d1 <= 0.0) {
    return b1;
  }
  if (seg_dist_m < d1) {
    return b1;
  }
  const double g = std::max(0.0, gap);
  if (seg_dist_m < d1 + g) {
    return 0.0;
  }
  const double d2_eff = (d2 > 0.0) ? d2 : d1;
  if (seg_dist_m < d1 + g + d2_eff) {
    return b2;
  }
  return 0.0;
}

// ============================================================
// 一张地图的分段计划
//
// 岔路 fork（3 段）：
//   段1 起步直道 → 段2 第 2 根黄线（选支/转弯）→ 段3 再过 D 米（恢复正常跟线）
// 环岛 roundabout（4 段）：
//   段1 → 段2（第2根黄线）→ 段3（第3根黄线）→ 段4（第3根黄线后 X 米，环岛二出口）
//
// 边界有两种，都靠计数/里程，不依赖视觉判据：
//   * 黄线计数  —— 黄线是精确锚点，且与停车逻辑共用同一个计数
//   * 段内里程  —— 没有黄线锚点的边界（选支完成、环岛出口）只能用里程
// 所有 *_count / *_dist_m 填 0 表示"该条件关闭"。
// ============================================================
struct SegmentPlan
{
  int segment_count{3};
  int seg2_start_count{2};          // 跨过第 N 根黄线 → 段2
  int seg3_start_count{0};          // 跨过第 N 根黄线 → 段3（0=不用黄线判定）
  double seg3_start_dist_m{0.0};    // 距上一根黄线再走 N 米 → 段3（0=不用里程判定）
  double seg4_start_dist_m{0.0};    // 距段3起点那根黄线再走 N 米 → 段4（0=无段4）
};

// ============================================================
// 分段判定：纯整数/浮点运算，不依赖 ROS、图像或时间。
// 段号在一次运行内单调递增（终点黄线会把里程清零，不单调就会倒回前一段），
// 只有 reset() 能把段号退回 1（换一圈时调用）。
// ============================================================
class SegmentPlanner
{
public:
  void configure(const SegmentPlan & plan);
  int update(int yellow_count, double dist_since_yellow);
  void reset();
  int current() const {return current_;}
  int count() const {return plan_.segment_count;}
  const SegmentPlan & plan() const {return plan_;}

private:
  SegmentPlan plan_{};
  int current_{1};
};

}  // namespace line_follower
