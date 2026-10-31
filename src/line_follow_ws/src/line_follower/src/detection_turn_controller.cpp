#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace line_follower
{

enum class DigitTaskState
{
  IDLE = 0,
  CANDIDATE,
  CONFIRMED,
  EXECUTING,
  RECOVERED,
  COOLDOWN
};

enum class TurnEndReason
{
  NONE = 0,
  VISUAL_RECOVERY,
  MAX_DURATION_TIMEOUT,
  ABORTED_BY_STOP_OR_DISARM
};

inline const char * digit_task_state_to_string(DigitTaskState state)
{
  switch (state) {
    case DigitTaskState::IDLE:
      return "idle";
    case DigitTaskState::CANDIDATE:
      return "candidate";
    case DigitTaskState::CONFIRMED:
      return "confirmed";
    case DigitTaskState::EXECUTING:
      return "executing";
    case DigitTaskState::RECOVERED:
      return "recovered";
    case DigitTaskState::COOLDOWN:
      return "cooldown";
    default:
      return "unknown";
  }
}

inline const char * turn_end_reason_to_string(TurnEndReason reason)
{
  switch (reason) {
    case TurnEndReason::NONE:
      return "none";
    case TurnEndReason::VISUAL_RECOVERY:
      return "visual_recovery";
    case TurnEndReason::MAX_DURATION_TIMEOUT:
      return "max_duration_timeout";
    case TurnEndReason::ABORTED_BY_STOP_OR_DISARM:
      return "aborted_by_stop_or_disarm";
    default:
      return "unknown";
  }
}

struct DetectionRoiInput
{
  std::string label;
  double confidence{0.0};
  int x_offset{0};
  int y_offset{0};
  int width{0};
  int height{0};
};

struct DetectionTurnConfig
{
  double confidence_threshold{0.6};
  int det_frame_width{640};
  int det_frame_height{480};
  int frame_window{3};
  double turn_cooldown{0.8};
  double turn_duration{0.6};
  double turn_linear{0.2};
  double yaw_gain{1.6};
  double max_yaw_rate{1.3};
  double y_align_tolerance{0.05};
  double turn_direction_sign{-1.0};
  double centroid_timeout{0.5};

  // 门控参数
  bool require_start_cmd{true};
  double stop_cmd_timeout{0.5};

  // 候选跟踪、连续性与综合评分参数（不按数字大小排序）
  double max_detection_gap{0.35};
  int max_missed_frames{1};
  double max_position_jump{0.18};
  double min_position_consistency{0.45};
  double min_x_diff{0.01};
  double weight_confidence{0.45};
  double weight_count{0.35};
  double weight_consistency{0.20};

  // 视觉恢复与去重参数
  double min_turn_duration{0.12};
  double recovery_center_tolerance{0.12};
  int recovery_confirm_frames{2};
  double dedup_clear_time{2.0};
};

struct CandidateTrack
{
  bool active{false};
  int digit{-1};
  int consecutive_count{0};
  int missed_frames{0};
  double mean_conf{0.0};
  double max_conf{0.0};
  double pos_consistency{0.0};
  double last_x_norm{0.5};
  double last_y_norm{0.5};
  double smoothed_x_diff{0.0};
  double smoothed_y_diff{0.0};
  double last_seen_time{-1e9};
  double composite_score{0.0};

  void reset()
  {
    *this = CandidateTrack{};
  }
};

struct TurnDecisionOutput
{
  DigitTaskState state{DigitTaskState::IDLE};
  bool should_publish{false};
  double linear_x{0.0};
  double angular_z{0.0};
  int active_digit{-1};
  double active_score{0.0};
  double active_conf{0.0};
  double active_x_diff{0.0};
  bool just_confirmed{false};
  bool just_recovered{false};
  TurnEndReason end_reason{TurnEndReason::NONE};
};

class DetectionTurnLogic
{
public:
  DetectionTurnLogic() = default;
  explicit DetectionTurnLogic(const DetectionTurnConfig & cfg)
  : cfg_(cfg)
  {
  }

  void set_config(const DetectionTurnConfig & cfg)
  {
    cfg_ = cfg;
  }

  const DetectionTurnConfig & config() const
  {
    return cfg_;
  }

  static int parse_digit_label(const std::string & label)
  {
    if (label.empty()) {
      return -1;
    }
    size_t start = (label[0] == 'd' || label[0] == 'D') ? 1 : 0;
    if (start >= label.size()) {
      return -1;
    }
    for (size_t i = start; i < label.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(label[i]))) {
        return -1;
      }
    }
    int value = -1;
    try {
      value = std::stoi(label.substr(start));
    } catch (...) {
      return -1;
    }
    if (value >= 1 && value <= 8) {
      return value;
    }
    return -1;
  }

  void update_start_cmd(bool started, double now_sec)
  {
    const bool prev_started = start_enabled_;
    start_enabled_ = started;
    has_seen_start_cmd_ = true;

    if (!started) {
      abort_for_stop_or_disarm(now_sec);
    } else if (!prev_started && started) {
      // 新一轮启动：清理去重锁与旧状态
      reset_all(now_sec);
      start_enabled_ = true;
      has_seen_start_cmd_ = true;
    }
  }

  void update_stop_cmd(double linear_x, double angular_z, double now_sec)
  {
    if (std::abs(linear_x) < 1e-4 && std::abs(angular_z) < 1e-4) {
      last_stop_cmd_time_ = now_sec;
      has_stop_cmd_ = true;
      abort_for_stop_or_disarm(now_sec);
    }
  }

  void update_track_state(const std::string & track_state, double now_sec)
  {
    track_state_ = track_state;
    if (track_state == "yellow_line_detected" ||
        track_state == "stopped" ||
        track_state == "idle")
    {
      abort_for_stop_or_disarm(now_sec);
      if (track_state == "yellow_line_detected") {
        // 黄线停靠后进入新赛段，解除上一赛段数字去重锁与冷却
        last_executed_digit_ = -1;
        last_executed_digit_seen_time_ = -1e9;
        cooldown_until_sec_ = now_sec;
      }
    }
  }

  void update_centroid(double norm_x, double norm_y, double now_sec)
  {
    last_centroid_norm_x_ = std::clamp(norm_x, 0.0, 1.0);
    last_centroid_norm_y_ = std::clamp(norm_y, 0.0, 1.0);
    last_centroid_time_ = now_sec;
    has_centroid_ = true;

    if (state_ == DigitTaskState::CONFIRMED || state_ == DigitTaskState::EXECUTING) {
      if (track_state_ == "tracking" &&
          std::abs(last_centroid_norm_x_ - 0.5) <= cfg_.recovery_center_tolerance)
      {
        recovery_centered_count_++;
      } else {
        recovery_centered_count_ = 0;
      }
    }
  }

  bool is_armed(double now_sec) const
  {
    if (cfg_.require_start_cmd && !start_enabled_) {
      return false;
    }
    if (track_state_ == "yellow_line_detected" ||
        track_state_ == "stopped" ||
        track_state_ == "idle")
    {
      return false;
    }
    if (has_stop_cmd_ && (now_sec - last_stop_cmd_time_) <= cfg_.stop_cmd_timeout) {
      return false;
    }
    return true;
  }

  TurnDecisionOutput on_detection(
    const std::vector<DetectionRoiInput> & rois,
    double now_sec)
  {
    TurnDecisionOutput out{};

    if (!is_armed(now_sec)) {
      abort_for_stop_or_disarm(now_sec);
      out.state = state_;
      out.end_reason = last_end_reason_;
      return out;
    }

    // 先推进可能已到期的状态转换
    advance_non_detection_states(now_sec);

    // 若已确认、正在执行转向、刚恢复或处于冷却期，不接受新候选触发
    if (state_ == DigitTaskState::CONFIRMED ||
        state_ == DigitTaskState::EXECUTING ||
        state_ == DigitTaskState::RECOVERED ||
        state_ == DigitTaskState::COOLDOWN)
    {
      // 但若画面中仍看到刚执行过的数字，刷新其可见时间，防止仍在画面中时去重锁过早解除
      refresh_executed_digit_sighting(rois, now_sec);
      return current_output();
    }

    // 检查去重锁是否因数字长时间消失而自动解除
    if (last_executed_digit_ > 0 &&
        (now_sec - last_executed_digit_seen_time_) > cfg_.dedup_clear_time)
    {
      last_executed_digit_ = -1;
    }

    const bool centroid_valid =
      has_centroid_ && ((now_sec - last_centroid_time_) <= cfg_.centroid_timeout);
    if (!centroid_valid) {
      expire_stale_tracks(now_sec, true);
      update_candidate_state();
      return current_output();
    }

    const double frame_w = std::max(1, cfg_.det_frame_width);
    const double frame_h = std::max(1, cfg_.det_frame_height);

    struct FrameObs
    {
      bool seen{false};
      double conf{0.0};
      double x_norm{0.5};
      double y_norm{0.5};
      double x_diff{0.0};
      double y_diff{0.0};
    };
    std::array<FrameObs, 9> frame_obs{};

    for (const auto & roi : rois) {
      if (roi.width <= 0 || roi.height <= 0 || roi.confidence < cfg_.confidence_threshold) {
        continue;
      }
      const int digit = parse_digit_label(roi.label);
      if (digit < 1 || digit > 8) {
        continue;
      }

      const double cx = static_cast<double>(roi.x_offset) + static_cast<double>(roi.width) * 0.5;
      const double cy = static_cast<double>(roi.y_offset) + static_cast<double>(roi.height) * 0.5;
      const double x_norm = std::clamp(cx / frame_w, 0.0, 1.0);
      const double y_norm = std::clamp(cy / frame_h, 0.0, 1.0);

      // 若是刚执行完的数字，刷新其可见时间戳并跳过（重复检测不重复转向）
      if (digit == last_executed_digit_) {
        last_executed_digit_seen_time_ = now_sec;
        continue;
      }

      const double y_diff = y_norm - last_centroid_norm_y_;
      if (std::abs(y_diff) > cfg_.y_align_tolerance) {
        continue;
      }

      const double x_diff = x_norm - last_centroid_norm_x_;
      if (std::abs(x_diff) < cfg_.min_x_diff) {
        continue;
      }

      auto & slot = frame_obs[static_cast<size_t>(digit)];
      if (!slot.seen || roi.confidence > slot.conf) {
        slot.seen = true;
        slot.conf = roi.confidence;
        slot.x_norm = x_norm;
        slot.y_norm = y_norm;
        slot.x_diff = x_diff;
        slot.y_diff = y_diff;
      }
    }

    const int req_frames = std::max(1, cfg_.frame_window);

    for (int d = 1; d <= 8; ++d) {
      auto & trk = tracks_[static_cast<size_t>(d)];
      const auto & obs = frame_obs[static_cast<size_t>(d)];

      if (obs.seen) {
        const double dt = now_sec - trk.last_seen_time;
        if (trk.active && dt <= cfg_.max_detection_gap && trk.missed_frames <= cfg_.max_missed_frames) {
          const double jump = std::hypot(obs.x_norm - trk.last_x_norm, obs.y_norm - trk.last_y_norm);
          if (jump <= cfg_.max_position_jump) {
            trk.consecutive_count++;
            trk.missed_frames = 0;
            trk.mean_conf =
              (trk.mean_conf * static_cast<double>(trk.consecutive_count - 1) + obs.conf) /
              static_cast<double>(trk.consecutive_count);
            trk.max_conf = std::max(trk.max_conf, obs.conf);
            const double step_cons =
              std::clamp(1.0 - jump / std::max(1e-3, cfg_.max_position_jump), 0.0, 1.0);
            trk.pos_consistency = 0.55 * trk.pos_consistency + 0.45 * step_cons;
            trk.last_x_norm = obs.x_norm;
            trk.last_y_norm = obs.y_norm;
            trk.smoothed_x_diff = 0.5 * trk.smoothed_x_diff + 0.5 * obs.x_diff;
            trk.smoothed_y_diff = 0.5 * trk.smoothed_y_diff + 0.5 * obs.y_diff;
            trk.last_seen_time = now_sec;
          } else {
            // 位置跳变过大：重置连续计数并降低初始位置一致性
            trk.active = true;
            trk.digit = d;
            trk.consecutive_count = 1;
            trk.missed_frames = 0;
            trk.mean_conf = obs.conf;
            trk.max_conf = obs.conf;
            trk.pos_consistency = 0.25;
            trk.last_x_norm = obs.x_norm;
            trk.last_y_norm = obs.y_norm;
            trk.smoothed_x_diff = obs.x_diff;
            trk.smoothed_y_diff = obs.y_diff;
            trk.last_seen_time = now_sec;
          }
        } else {
          trk.active = true;
          trk.digit = d;
          trk.consecutive_count = 1;
          trk.missed_frames = 0;
          trk.mean_conf = obs.conf;
          trk.max_conf = obs.conf;
          trk.pos_consistency = (req_frames <= 1) ? 0.85 : 0.75;
          trk.last_x_norm = obs.x_norm;
          trk.last_y_norm = obs.y_norm;
          trk.smoothed_x_diff = obs.x_diff;
          trk.smoothed_y_diff = obs.y_diff;
          trk.last_seen_time = now_sec;
        }

        const double count_score = std::clamp(
          static_cast<double>(trk.consecutive_count) / static_cast<double>(req_frames),
          0.0,
          1.25);
        const double conf_score = 0.65 * trk.mean_conf + 0.35 * trk.max_conf;
        trk.composite_score =
          cfg_.weight_confidence * conf_score +
          cfg_.weight_count * count_score +
          cfg_.weight_consistency * trk.pos_consistency;
      } else if (trk.active) {
        trk.missed_frames++;
        if ((now_sec - trk.last_seen_time) > cfg_.max_detection_gap ||
            trk.missed_frames > cfg_.max_missed_frames)
        {
          trk.reset();
        }
      }
    }

    // 筛选综合评分最高的候选（绝不按数字大小排序）
    int best_confirmed_digit = -1;
    double best_confirmed_score = -1.0;
    int best_candidate_digit = -1;
    double best_candidate_score = -1.0;

    for (int d = 1; d <= 8; ++d) {
      const auto & trk = tracks_[static_cast<size_t>(d)];
      if (!trk.active) {
        continue;
      }

      if (trk.composite_score > best_candidate_score ||
          (std::abs(trk.composite_score - best_candidate_score) <= 1e-9 &&
           trk.mean_conf > tracks_[static_cast<size_t>(std::max(1, best_candidate_digit))].mean_conf))
      {
        best_candidate_score = trk.composite_score;
        best_candidate_digit = d;
      }

      const bool meets_confirm =
        trk.consecutive_count >= req_frames &&
        trk.mean_conf >= cfg_.confidence_threshold &&
        trk.pos_consistency >= cfg_.min_position_consistency;

      if (meets_confirm) {
        if (trk.composite_score > best_confirmed_score ||
            (std::abs(trk.composite_score - best_confirmed_score) <= 1e-9 &&
             trk.mean_conf > tracks_[static_cast<size_t>(std::max(1, best_confirmed_digit))].mean_conf))
        {
          best_confirmed_score = trk.composite_score;
          best_confirmed_digit = d;
        }
      }
    }

    if (best_confirmed_digit > 0) {
      const auto & chosen = tracks_[static_cast<size_t>(best_confirmed_digit)];
      const double yaw = std::clamp(
        cfg_.turn_direction_sign * chosen.smoothed_x_diff * cfg_.yaw_gain,
        -cfg_.max_yaw_rate,
        cfg_.max_yaw_rate);

      if (std::abs(yaw) > 1e-6) {
        state_ = DigitTaskState::CONFIRMED;
        active_digit_ = chosen.digit;
        active_score_ = chosen.composite_score;
        active_conf_ = chosen.mean_conf;
        active_x_diff_ = chosen.smoothed_x_diff;
        turn_linear_cmd_ = cfg_.turn_linear;
        turn_yaw_cmd_ = yaw;
        turn_start_sec_ = now_sec;
        turn_max_end_sec_ = now_sec + std::max(0.05, cfg_.turn_duration);
        recovery_centered_count_ = 0;
        last_end_reason_ = TurnEndReason::NONE;
        clear_all_tracks();

        TurnDecisionOutput confirmed_out = current_output();
        confirmed_out.just_confirmed = true;
        return confirmed_out;
      }
    }

    if (best_candidate_digit > 0) {
      const auto & cand = tracks_[static_cast<size_t>(best_candidate_digit)];
      state_ = DigitTaskState::CANDIDATE;
      active_digit_ = cand.digit;
      active_score_ = cand.composite_score;
      active_conf_ = cand.mean_conf;
      active_x_diff_ = cand.smoothed_x_diff;
    } else {
      state_ = DigitTaskState::IDLE;
      active_digit_ = -1;
      active_score_ = 0.0;
      active_conf_ = 0.0;
      active_x_diff_ = 0.0;
    }

    return current_output();
  }

  TurnDecisionOutput step_control(double now_sec)
  {
    if (!is_armed(now_sec)) {
      abort_for_stop_or_disarm(now_sec);
      TurnDecisionOutput out = current_output();
      out.end_reason = last_end_reason_;
      return out;
    }

    if (state_ == DigitTaskState::CONFIRMED) {
      state_ = DigitTaskState::EXECUTING;
    }

    if (state_ == DigitTaskState::EXECUTING) {
      const double elapsed = now_sec - turn_start_sec_;
      const bool min_dur_ok = elapsed >= cfg_.min_turn_duration;
      const bool centroid_fresh =
        has_centroid_ && ((now_sec - last_centroid_time_) <= cfg_.centroid_timeout);
      const bool visual_recovered =
        min_dur_ok &&
        centroid_fresh &&
        track_state_ == "tracking" &&
        recovery_centered_count_ >= std::max(1, cfg_.recovery_confirm_frames);

      if (visual_recovered || now_sec >= turn_max_end_sec_) {
        state_ = DigitTaskState::RECOVERED;
        last_end_reason_ = visual_recovered ?
          TurnEndReason::VISUAL_RECOVERY :
          TurnEndReason::MAX_DURATION_TIMEOUT;
        last_executed_digit_ = active_digit_;
        last_executed_digit_seen_time_ = now_sec;
        cooldown_until_sec_ = now_sec + std::max(0.0, cfg_.turn_cooldown);
        turn_linear_cmd_ = 0.0;
        turn_yaw_cmd_ = 0.0;

        TurnDecisionOutput rec_out = current_output();
        rec_out.just_recovered = true;
        rec_out.end_reason = last_end_reason_;
        return rec_out;
      }

      return current_output();
    }

    if (state_ == DigitTaskState::RECOVERED) {
      if (now_sec < cooldown_until_sec_) {
        state_ = DigitTaskState::COOLDOWN;
      } else {
        state_ = DigitTaskState::IDLE;
      }
      return current_output();
    }

    if (state_ == DigitTaskState::COOLDOWN) {
      if (now_sec >= cooldown_until_sec_) {
        state_ = DigitTaskState::IDLE;
        active_digit_ = -1;
        active_score_ = 0.0;
        active_conf_ = 0.0;
        active_x_diff_ = 0.0;
      }
      return current_output();
    }

    if (state_ == DigitTaskState::CANDIDATE) {
      expire_stale_tracks(now_sec, false);
      update_candidate_state();
    }

    return current_output();
  }

  DigitTaskState state() const
  {
    return state_;
  }

  int last_executed_digit() const
  {
    return last_executed_digit_;
  }

  TurnEndReason last_end_reason() const
  {
    return last_end_reason_;
  }

  const CandidateTrack & track(int digit) const
  {
    return tracks_[static_cast<size_t>(std::clamp(digit, 0, 8))];
  }

  void reset_all(double now_sec)
  {
    clear_all_tracks();
    state_ = DigitTaskState::IDLE;
    active_digit_ = -1;
    active_score_ = 0.0;
    active_conf_ = 0.0;
    active_x_diff_ = 0.0;
    turn_linear_cmd_ = 0.0;
    turn_yaw_cmd_ = 0.0;
    turn_start_sec_ = now_sec;
    turn_max_end_sec_ = now_sec;
    cooldown_until_sec_ = now_sec;
    last_executed_digit_ = -1;
    last_executed_digit_seen_time_ = -1e9;
    recovery_centered_count_ = 0;
    last_end_reason_ = TurnEndReason::NONE;
  }

private:
  void clear_all_tracks()
  {
    for (auto & t : tracks_) {
      t.reset();
    }
  }

  void expire_stale_tracks(double now_sec, bool force_clear)
  {
    for (int d = 1; d <= 8; ++d) {
      auto & trk = tracks_[static_cast<size_t>(d)];
      if (!trk.active) {
        continue;
      }
      if (force_clear || (now_sec - trk.last_seen_time) > cfg_.max_detection_gap) {
        trk.reset();
      }
    }
  }

  void update_candidate_state()
  {
    int best_d = -1;
    double best_s = -1.0;
    for (int d = 1; d <= 8; ++d) {
      const auto & trk = tracks_[static_cast<size_t>(d)];
      if (trk.active && trk.composite_score > best_s) {
        best_s = trk.composite_score;
        best_d = d;
      }
    }
    if (best_d > 0) {
      state_ = DigitTaskState::CANDIDATE;
      active_digit_ = best_d;
      active_score_ = tracks_[static_cast<size_t>(best_d)].composite_score;
      active_conf_ = tracks_[static_cast<size_t>(best_d)].mean_conf;
      active_x_diff_ = tracks_[static_cast<size_t>(best_d)].smoothed_x_diff;
    } else if (state_ == DigitTaskState::CANDIDATE) {
      state_ = DigitTaskState::IDLE;
      active_digit_ = -1;
      active_score_ = 0.0;
      active_conf_ = 0.0;
      active_x_diff_ = 0.0;
    }
  }

  void advance_non_detection_states(double now_sec)
  {
    if (state_ == DigitTaskState::RECOVERED) {
      state_ = (now_sec < cooldown_until_sec_) ? DigitTaskState::COOLDOWN : DigitTaskState::IDLE;
    }
    if (state_ == DigitTaskState::COOLDOWN && now_sec >= cooldown_until_sec_) {
      state_ = DigitTaskState::IDLE;
      active_digit_ = -1;
      active_score_ = 0.0;
      active_conf_ = 0.0;
      active_x_diff_ = 0.0;
    }
  }

  void refresh_executed_digit_sighting(
    const std::vector<DetectionRoiInput> & rois,
    double now_sec)
  {
    if (last_executed_digit_ < 1) {
      return;
    }
    for (const auto & roi : rois) {
      if (roi.width <= 0 || roi.height <= 0 || roi.confidence < cfg_.confidence_threshold) {
        continue;
      }
      if (parse_digit_label(roi.label) == last_executed_digit_) {
        last_executed_digit_seen_time_ = now_sec;
        break;
      }
    }
  }

  void abort_for_stop_or_disarm(double now_sec)
  {
    clear_all_tracks();
    if (state_ == DigitTaskState::CONFIRMED || state_ == DigitTaskState::EXECUTING) {
      last_end_reason_ = TurnEndReason::ABORTED_BY_STOP_OR_DISARM;
    }
    if (state_ != DigitTaskState::COOLDOWN || now_sec >= cooldown_until_sec_) {
      state_ = DigitTaskState::IDLE;
    }
    turn_linear_cmd_ = 0.0;
    turn_yaw_cmd_ = 0.0;
    active_digit_ = -1;
    active_score_ = 0.0;
    active_conf_ = 0.0;
    active_x_diff_ = 0.0;
    recovery_centered_count_ = 0;
  }

  TurnDecisionOutput current_output() const
  {
    TurnDecisionOutput out{};
    out.state = state_;
    const bool publishing =
      (state_ == DigitTaskState::CONFIRMED || state_ == DigitTaskState::EXECUTING);
    out.should_publish = publishing;
    out.linear_x = publishing ? turn_linear_cmd_ : 0.0;
    out.angular_z = publishing ? turn_yaw_cmd_ : 0.0;
    out.active_digit = active_digit_;
    out.active_score = active_score_;
    out.active_conf = active_conf_;
    out.active_x_diff = active_x_diff_;
    out.end_reason = last_end_reason_;
    return out;
  }

  DetectionTurnConfig cfg_{};
  std::array<CandidateTrack, 9> tracks_{};

  DigitTaskState state_{DigitTaskState::IDLE};
  int active_digit_{-1};
  double active_score_{0.0};
  double active_conf_{0.0};
  double active_x_diff_{0.0};

  double turn_linear_cmd_{0.0};
  double turn_yaw_cmd_{0.0};
  double turn_start_sec_{0.0};
  double turn_max_end_sec_{0.0};
  double cooldown_until_sec_{0.0};

  int last_executed_digit_{-1};
  double last_executed_digit_seen_time_{-1e9};
  int recovery_centered_count_{0};
  TurnEndReason last_end_reason_{TurnEndReason::NONE};

  bool start_enabled_{false};
  bool has_seen_start_cmd_{false};
  std::string track_state_{"tracking"};
  bool has_stop_cmd_{false};
  double last_stop_cmd_time_{-1e9};

  bool has_centroid_{false};
  double last_centroid_norm_x_{0.5};
  double last_centroid_norm_y_{0.5};
  double last_centroid_time_{-1e9};
};

}  // namespace line_follower

#ifndef DETECTION_TURN_CONTROLLER_PURE_LOGIC_ONLY

#include <ai_msgs/msg/perception_targets.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

class DetectionTurnController : public rclcpp::Node
{
public:
  DetectionTurnController()
  : Node("detection_turn_controller")
  {
    detection_topic_ =
      this->declare_parameter<std::string>("detection_topic", "hobot_dnn_detection");
    state_topic_ =
      this->declare_parameter<std::string>("state_topic", "line_follower/track_state");
    cmd_vel_topic_ =
      this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_detection");
    line_centroid_topic_ =
      this->declare_parameter<std::string>("line_centroid_topic", "line_follower/centroid");
    start_topic_ =
      this->declare_parameter<std::string>("start_topic", "line_follower/start_cmd");
    stop_cmd_topic_ =
      this->declare_parameter<std::string>("stop_cmd_topic", "/cmd_vel_stop");

    line_follower::DetectionTurnConfig cfg{};
    cfg.confidence_threshold =
      this->declare_parameter<double>("confidence_threshold", 0.6);
    cfg.det_frame_width =
      this->declare_parameter<int>("det_frame_width", 640);
    cfg.det_frame_height =
      this->declare_parameter<int>("det_frame_height", 480);
    cfg.frame_window =
      this->declare_parameter<int>("frame_window", 3);
    cfg.turn_cooldown =
      this->declare_parameter<double>("turn_cooldown", 0.8);
    cfg.turn_duration =
      this->declare_parameter<double>("turn_duration", 0.6);
    cfg.turn_linear =
      this->declare_parameter<double>("turn_linear", 0.3);
    cfg.yaw_gain =
      this->declare_parameter<double>("yaw_gain", 1.0);
    cfg.max_yaw_rate =
      this->declare_parameter<double>("max_yaw_rate", 1.2);
    cfg.y_align_tolerance =
      this->declare_parameter<double>("y_align_tolerance", 0.05);
    cfg.turn_direction_sign =
      this->declare_parameter<double>("turn_direction_sign", -1.0);
    cfg.centroid_timeout =
      this->declare_parameter<double>("centroid_timeout", 0.5);

    cfg.require_start_cmd =
      this->declare_parameter<bool>("require_start_cmd", true);
    cfg.stop_cmd_timeout =
      this->declare_parameter<double>("stop_cmd_timeout", 0.5);

    cfg.max_detection_gap =
      this->declare_parameter<double>("max_detection_gap", 0.35);
    cfg.max_missed_frames =
      this->declare_parameter<int>("max_missed_frames", 1);
    cfg.max_position_jump =
      this->declare_parameter<double>("max_position_jump", 0.18);
    cfg.min_position_consistency =
      this->declare_parameter<double>("min_position_consistency", 0.45);
    cfg.min_x_diff =
      this->declare_parameter<double>("min_x_diff", 0.01);
    cfg.weight_confidence =
      this->declare_parameter<double>("weight_confidence", 0.45);
    cfg.weight_count =
      this->declare_parameter<double>("weight_count", 0.35);
    cfg.weight_consistency =
      this->declare_parameter<double>("weight_consistency", 0.20);

    cfg.min_turn_duration =
      this->declare_parameter<double>("min_turn_duration", 0.12);
    cfg.recovery_center_tolerance =
      this->declare_parameter<double>("recovery_center_tolerance", 0.12);
    cfg.recovery_confirm_frames =
      this->declare_parameter<int>("recovery_confirm_frames", 2);
    cfg.dedup_clear_time =
      this->declare_parameter<double>("dedup_clear_time", 2.0);

    logic_.set_config(cfg);

    detection_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&DetectionTurnController::detection_callback, this, std::placeholders::_1));

    centroid_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      line_centroid_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::centroid_callback, this, std::placeholders::_1));

    state_sub_ = this->create_subscription<std_msgs::msg::String>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::state_callback, this, std::placeholders::_1));

    start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      start_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::start_callback, this, std::placeholders::_1));

    stop_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      stop_cmd_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::stop_callback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&DetectionTurnController::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Detection turn controller listening detections '%s', centroid '%s', state '%s', start '%s', publishing '%s'",
      detection_topic_.c_str(),
      line_centroid_topic_.c_str(),
      state_topic_.c_str(),
      start_topic_.c_str(),
      cmd_vel_topic_.c_str());
  }

private:
  void start_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    logic_.update_start_cmd(msg->data, this->now().seconds());
  }

  void stop_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    logic_.update_stop_cmd(msg->linear.x, msg->angular.z, this->now().seconds());
  }

  void centroid_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    logic_.update_centroid(
      static_cast<double>(msg->x),
      static_cast<double>(msg->y),
      this->now().seconds());
  }

  void state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    logic_.update_track_state(msg->data, this->now().seconds());
  }

  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const double now_sec = this->now().seconds();

    std::vector<line_follower::DetectionRoiInput> rois;
    for (const auto & target : msg->targets) {
      for (const auto & roi : target.rois) {
        rois.push_back(line_follower::DetectionRoiInput{
          roi.type,
          static_cast<double>(roi.confidence),
          roi.rect.x_offset,
          roi.rect.y_offset,
          roi.rect.width,
          roi.rect.height});
      }
    }

    const auto out = logic_.on_detection(rois, now_sec);
    if (out.just_confirmed) {
      RCLCPP_INFO(
        get_logger(),
        "[DEC] digit turn confirmed digit=%d score=%.2f conf=%.2f xdiff=%+.2f yaw=%+.2f max_dur=%.2f lin=%.2f",
        out.active_digit,
        out.active_score,
        out.active_conf,
        out.active_x_diff,
        out.angular_z,
        logic_.config().turn_duration,
        out.linear_x);
    }

    if (out.should_publish) {
      geometry_msgs::msg::Twist cmd;
      cmd.linear.x = out.linear_x;
      cmd.angular.z = out.angular_z;
      cmd_pub_->publish(cmd);
    }
  }

  void on_timer()
  {
    const double now_sec = this->now().seconds();
    const auto out = logic_.step_control(now_sec);

    if (out.just_recovered) {
      RCLCPP_INFO(
        get_logger(),
        "[DEC] digit turn finished digit=%d state=%s reason=%s",
        logic_.last_executed_digit(),
        line_follower::digit_task_state_to_string(out.state),
        line_follower::turn_end_reason_to_string(out.end_reason));
    }

    if (out.should_publish) {
      geometry_msgs::msg::Twist cmd;
      cmd.linear.x = out.linear_x;
      cmd.angular.z = out.angular_z;
      cmd_pub_->publish(cmd);
    }
  }

  std::string detection_topic_;
  std::string state_topic_;
  std::string cmd_vel_topic_;
  std::string line_centroid_topic_;
  std::string start_topic_;
  std::string stop_cmd_topic_;

  line_follower::DetectionTurnLogic logic_{};

  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr centroid_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr stop_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionTurnController>());
  rclcpp::shutdown();
  return 0;
}

#endif  // DETECTION_TURN_CONTROLLER_PURE_LOGIC_ONLY
