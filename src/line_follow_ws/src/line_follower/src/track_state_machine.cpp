#include "line_follower/track_state_machine.hpp"

#include <algorithm>
#include <cmath>

namespace line_follower
{

TrackStateMachine::TrackStateMachine()
{
  set_config(TrackStateConfig{});
}

TrackStateMachine::TrackStateMachine(const TrackStateConfig & config)
{
  set_config(config);
}

void TrackStateMachine::set_config(const TrackStateConfig & config)
{
  config_ = config;
  config_.yellow_permanent_count = std::max(1, config_.yellow_permanent_count);
  config_.yellow_clear_frames_required = std::max(1, config_.yellow_clear_frames_required);
  config_.yellow_stop_duration = std::max(0.0, config_.yellow_stop_duration);
  config_.yellow_stop_min_time = std::max(0.0, config_.yellow_stop_min_time);
  config_.odom_still_threshold = std::max(0.0, config_.odom_still_threshold);
  config_.odom_still_hold = std::max(0.0, config_.odom_still_hold);
  config_.line_lost_stop_time = std::max(0.0, config_.line_lost_stop_time);
}

const TrackStateConfig & TrackStateMachine::config() const
{
  return config_;
}

void TrackStateMachine::reset(double now_sec)
{
  const double prev_odom_speed = state_.odom_speed;
  const bool prev_odom_still_valid = state_.odom_still_valid;
  const bool prev_odom_seen = state_.odom_seen;
  const double prev_odom_still_since = state_.odom_still_since_sec;

  state_ = TrackRuntimeState{};
  state_.yellow_line_detection_time_sec = now_sec;
  state_.last_detection_time_sec = now_sec;
  state_.last_state = TrackState::LOST;

  state_.odom_speed = prev_odom_speed;
  state_.odom_still_valid = prev_odom_still_valid;
  state_.odom_seen = prev_odom_seen;
  state_.odom_still_since_sec = prev_odom_still_since;
}

bool TrackStateMachine::on_start_command(bool old_enabled, bool new_enabled, double now_sec)
{
  if (!old_enabled && new_enabled && state_.permanent_stop) {
    reset(now_sec);
    return true;
  }
  return false;
}

void TrackStateMachine::update_odom(double linear_x, double now_sec)
{
  state_.odom_speed = std::abs(linear_x);
  state_.odom_seen = true;
  if (state_.odom_speed < config_.odom_still_threshold) {
    if (!state_.odom_still_valid) {
      state_.odom_still_valid = true;
      state_.odom_still_since_sec = now_sec;
    }
  } else {
    state_.odom_still_valid = false;
  }
}

TrackStepEvents TrackStateMachine::step(
  bool yellow_detected,
  bool line_detected,
  double now_sec)
{
  TrackStepEvents events{};

  if (yellow_detected) {
    state_.yellow_clear_frames = 0;

    if (!state_.yellow_line_latched) {
      state_.yellow_line_latched = true;
      state_.yellow_line_detection_count++;
      state_.yellow_line_detection_time_sec = now_sec;
      events.yellow_stop_begin = true;

      if (state_.yellow_line_detection_count >= config_.yellow_permanent_count) {
        state_.permanent_stop = true;
        state_.yellow_line_stop_active = false;
        events.yellow_final_stop = true;
      } else {
        state_.yellow_line_stop_active = true;
        events.yellow_temp_stop = true;
      }
    }
  } else if (state_.yellow_line_latched) {
    state_.yellow_clear_frames++;

    if (state_.yellow_clear_frames >= config_.yellow_clear_frames_required) {
      state_.yellow_line_latched = false;
      state_.yellow_clear_frames = 0;
      events.yellow_cleared = true;
    }
  }

  events.since_yellow_detection = now_sec - state_.yellow_line_detection_time_sec;
  events.odom_still_ok =
    state_.odom_still_valid &&
    (now_sec - state_.odom_still_since_sec) >= config_.odom_still_hold;
  events.can_release =
    (events.since_yellow_detection >= config_.yellow_stop_min_time && events.odom_still_ok) ||
    (events.since_yellow_detection >= config_.yellow_stop_duration);

  if (state_.permanent_stop) {
    events.yellow_hold_stop = true;
    events.base_state = TrackState::YELLOW_LINE_DETECTED;
  } else if (state_.yellow_line_stop_active && !events.can_release) {
    events.yellow_hold_stop = true;
    events.base_state = TrackState::YELLOW_LINE_DETECTED;
  } else {
    if (state_.yellow_line_stop_active && events.can_release) {
      state_.yellow_line_stop_active = false;
      events.yellow_stop_end = true;
    }

    if (line_detected) {
      state_.last_detection_time_sec = now_sec;
      events.base_state = TrackState::TRACKING;
    } else {
      events.since_line_detection = now_sec - state_.last_detection_time_sec;
      events.base_state =
        (events.since_line_detection >= config_.line_lost_stop_time) ?
        TrackState::LOST :
        state_.last_state;
    }
  }

  events.yellow_stop_now =
    state_.permanent_stop ||
    (state_.yellow_line_stop_active && events.since_yellow_detection < config_.yellow_stop_duration);

  events.yellow_hold_stop =
    state_.permanent_stop ||
    (state_.yellow_line_stop_active && !events.can_release);

  return events;
}

void TrackStateMachine::commit_state(TrackState final_state)
{
  state_.last_state = final_state;
}

const TrackRuntimeState & TrackStateMachine::state() const
{
  return state_;
}

TrackRuntimeState & TrackStateMachine::mutable_state()
{
  return state_;
}

}  // namespace line_follower
