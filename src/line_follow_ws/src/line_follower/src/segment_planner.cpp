#include "line_follower/segment_planner.hpp"

namespace line_follower
{

void SegmentPlanner::configure(const SegmentPlan & plan)
{
  plan_ = plan;
  plan_.segment_count = std::clamp(plan_.segment_count, 1, kMaxSegments);
  // 计数/里程填 0 表示"该条件关闭"，负数没有意义。
  plan_.seg2_start_count = std::max(0, plan_.seg2_start_count);
  plan_.seg3_start_count = std::max(0, plan_.seg3_start_count);
  plan_.seg3_start_dist_m = std::max(0.0, plan_.seg3_start_dist_m);
  plan_.seg4_start_dist_m = std::max(0.0, plan_.seg4_start_dist_m);
  current_ = 1;
}

void SegmentPlanner::reset()
{
  current_ = 1;
}

int SegmentPlanner::update(int yellow_count, double dist_since_yellow)
{
  int want = 1;

  if (plan_.seg2_start_count > 0 && yellow_count >= plan_.seg2_start_count) {
    want = 2;
  }

  // 段3 用两种边界之一：跨过第 N 根黄线（环岛），或从上一根黄线起再走 D 米（岔路选支）。
  const bool by_count =
    plan_.seg3_start_count > 0 && yellow_count >= plan_.seg3_start_count;
  const bool by_dist =
    plan_.seg3_start_dist_m > 0.0 && want >= 2 &&
    dist_since_yellow >= plan_.seg3_start_dist_m;

  if (plan_.segment_count >= 3 && (by_count || by_dist)) {
    want = 3;
    // 段4 与段3 共用起点，靠里程区分（环岛二出口）。
    if (plan_.segment_count >= 4 && plan_.seg4_start_dist_m > 0.0 &&
      dist_since_yellow >= plan_.seg4_start_dist_m)
    {
      want = 4;
    }
  }

  if (want > current_) {
    current_ = want;
  }
  return current_;
}

}  // namespace line_follower
