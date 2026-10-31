#pragma once

#include "line_follower/line_follower_types.hpp"

namespace line_follower
{

struct TrackStateConfig
{
  double yellow_stop_duration{1.0};
  int yellow_permanent_count{5};
  int yellow_clear_frames_required{10};

  double yellow_stop_min_time{0.35};
  double odom_still_threshold{0.02};
  double odom_still_hold{0.25};

  double line_lost_stop_time{0.8};
};

struct TrackRuntimeState
{
  bool yellow_line_latched{false};
  int yellow_clear_frames{0};
  bool yellow_line_stop_active{false};
  int yellow_line_detection_count{0};
  bool permanent_stop{false};
  double yellow_line_detection_time_sec{0.0};

  double odom_speed{0.0};
  bool odom_still_valid{false};
  bool odom_seen{false};
  double odom_still_since_sec{0.0};

  TrackState last_state{TrackState::LOST};
  double last_detection_time_sec{0.0};
};

struct TrackStepEvents
{
  bool yellow_stop_begin{false};
  bool yellow_final_stop{false};
  bool yellow_temp_stop{false};
  bool yellow_cleared{false};
  bool yellow_stop_end{false};

  double since_yellow_detection{0.0};
  bool odom_still_ok{false};
  bool can_release{false};
  bool yellow_stop_now{false};
  bool yellow_hold_stop{false};

  double since_line_detection{0.0};
  TrackState base_state{TrackState::LOST};
};

class TrackStateMachine
{
public:
  TrackStateMachine();
  explicit TrackStateMachine(const TrackStateConfig & config);

  void set_config(const TrackStateConfig & config);
  const TrackStateConfig & config() const;

  void reset(double now_sec);
  bool on_start_command(bool old_enabled, bool new_enabled, double now_sec);
  void update_odom(double linear_x, double now_sec);

  TrackStepEvents step(bool yellow_detected, bool line_detected, double now_sec);
  void commit_state(TrackState final_state);

  const TrackRuntimeState & state() const;
  TrackRuntimeState & mutable_state();

private:
  TrackStateConfig config_{};
  TrackRuntimeState state_{};
};

}  // namespace line_follower
