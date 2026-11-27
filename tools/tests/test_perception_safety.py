#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone unit tests for camera V4L2 policy, BPU YOLOv8 post-processing/tracker,
safety deadman watchdog, and launch/EKF configuration hygiene.

Runnable directly on Windows without ROS2 or Horizon BPU SDK:
    python tools/tests/test_perception_safety.py
"""

from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np

WS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS_ROOT / 'src' / 'camera'))
sys.path.insert(0, str(WS_ROOT / 'src' / 'yolov8'))
sys.path.insert(0, str(WS_ROOT / 'tools'))

import camera as camera_mod  # noqa: E402
import bpu_yolov8_node as yolo_mod  # noqa: E402
import safety_node as safety_mod  # noqa: E402


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeOutputProps:
    def __init__(self, shape, scale_data=None):
        self.shape = shape
        self.scale_data = scale_data if scale_data is not None else []


class FakeBPUOutput:
    def __init__(self, arr, scale=1.0):
        self.buffer = arr
        self.properties = FakeOutputProps(arr.shape, [scale] if scale is not None else [])


class FakeBPUModel:
    def __init__(self, shapes):
        self.outputs = [types.SimpleNamespace(properties=FakeOutputProps(s)) for s in shapes]


def _make_grids(model_w=640, model_h=640, strides=(8, 16, 32)):
    grids = []
    for s in strides:
        gw = model_w // s
        gh = model_h // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32) + 0.5
        grids.append((grid, s))
    return grids


def _build_synthetic_outputs(num_classes, injections, layout='NHWC'):
    """
    Create 3 cls heads + 3 reg heads (strides 8, 16, 32) with background logits = -8.0.
    injections: list of dicts {stride_idx, gy, gx, cls_id, logit, dfl_dist=(l,t,r,b)}
    """
    grid_sizes = [80, 40, 20]
    outputs = []
    for s_idx, g in enumerate(grid_sizes):
        cls_arr = np.full((1, g, g, num_classes), -8.0, dtype=np.float32)
        reg_arr = np.zeros((1, g, g, 64), dtype=np.float32)
        for inj in injections:
            if inj['stride_idx'] != s_idx:
                continue
            gy, gx = inj['gy'], inj['gx']
            cls_arr[0, gy, gx, inj['cls_id']] = float(inj['logit'])
            l, t, r, b = inj.get('dfl_dist', (2, 2, 2, 2))
            for side_idx, bin_val in enumerate((l, t, r, b)):
                base = side_idx * 16
                reg_arr[0, gy, gx, base:base + 16] = -10.0
                reg_arr[0, gy, gx, base + int(bin_val)] = 10.0
        if layout == 'NCHW':
            cls_arr = np.transpose(cls_arr, (0, 3, 1, 2)).copy()
            reg_arr = np.transpose(reg_arr, (0, 3, 1, 2)).copy()
        outputs.append(FakeBPUOutput(cls_arr, scale=1.0))
        outputs.append(FakeBPUOutput(reg_arr, scale=1.0))
    return outputs


class TestCameraV4L2Policy(unittest.TestCase):
    def test_auto_mode_startup_sets_anti_flicker_and_auto_controls(self):
        calls = []
        logs = []

        def fake_runner(cmd, **_kwargs):
            calls.append(cmd)
            return FakeCompletedProcess(0, '', '')

        policy = camera_mod.CameraV4L2Policy(
            device='/dev/video0',
            power_line_frequency=1,
            ae_awb_mode='auto',
            backlight_compensation=0,
            runner=fake_runner,
            logger=logs.append,
        )
        summary = policy.apply_startup()
        self.assertEqual(summary['power_line_frequency'], 1)
        self.assertEqual(summary['backlight_compensation'], 0)
        self.assertFalse(policy.locked)
        joined = [' '.join(c) for c in calls]
        self.assertTrue(any('power_line_frequency=1' in s for s in joined))
        self.assertTrue(any('backlight_compensation=0' in s for s in joined))
        self.assertTrue(any('exposure_dynamic_framerate=0' in s for s in joined))
        self.assertTrue(any('auto_exposure=3' in s for s in joined))
        self.assertTrue(any('white_balance_automatic=1' in s for s in joined))
        self.assertTrue(any('[DEC] camera v4l2 startup' in m for m in logs))

    def test_camera_30fps_buffer_and_dynamic_framerate_alias_fallback(self):
        calls = []

        def alias_runner(cmd, **_kwargs):
            calls.append(cmd)
            if '--set-ctrl' in cmd:
                val = cmd[cmd.index('--set-ctrl') + 1]
                if val.startswith('exposure_dynamic_framerate='):
                    return FakeCompletedProcess(1, '', 'unknown control')
            return FakeCompletedProcess(0, '', '')

        policy = camera_mod.CameraV4L2Policy(
            device='/dev/video0',
            ae_awb_mode='lock_on_start',
            exposure_dynamic_framerate=0,
            runner=alias_runner,
        )
        summary = policy.apply_startup()
        self.assertEqual(summary['exposure_dynamic_framerate'], 0)
        joined = [' '.join(c) for c in calls]
        self.assertTrue(any('exposure_auto_priority=0' in s for s in joined))

        cam_src = (WS_ROOT / 'src' / 'camera' / 'camera.py').read_text(encoding='utf-8')
        self.assertIn("declare_parameter('v4l2_buffer_size', 2)", cam_src)
        self.assertIn("declare_parameter('exposure_dynamic_framerate', 0)", cam_src)
        self.assertLess(
            cam_src.index('CAP_PROP_FOURCC'),
            cam_src.index('CAP_PROP_BUFFERSIZE'),
        )

    def test_unsupported_optional_ctrl_is_not_reported_as_set_fail(self):
        logs = []

        def unsupported_runner(cmd, **_kwargs):
            if '--set-ctrl' in cmd:
                ctrl = cmd[cmd.index('--set-ctrl') + 1]
                if ctrl.startswith(('exposure_dynamic_framerate=', 'exposure_auto_priority=')):
                    return FakeCompletedProcess(1, '', "unknown control 'exposure_auto_priority'")
            return FakeCompletedProcess(0, '', '')

        policy = camera_mod.CameraV4L2Policy(
            device='/dev/video0',
            ae_awb_mode='lock_on_start',
            exposure_dynamic_framerate=0,
            runner=unsupported_runner,
            logger=logs.append,
        )
        summary = policy.apply_startup()
        self.assertIsNone(summary['exposure_dynamic_framerate'])
        self.assertFalse(any('set_fail' in m for m in logs))
        self.assertTrue(any('unsupported_optional ctrl=exposure_dynamic_framerate' in m for m in logs))

    def test_manual_fixed_mode_locks_configured_values(self):
        calls = []

        def fake_runner(cmd, **_kwargs):
            calls.append(cmd)
            return FakeCompletedProcess(0, '', '')

        policy = camera_mod.CameraV4L2Policy(
            device='/dev/video2',
            power_line_frequency=1,
            ae_awb_mode='manual_fixed',
            exposure_time_absolute=80,
            white_balance_temperature=4600,
            gain=12,
            backlight_compensation=0,
            runner=fake_runner,
        )
        policy.apply_startup()
        self.assertTrue(policy.locked)
        joined = [' '.join(c) for c in calls]
        self.assertTrue(any('auto_exposure=1' in s for s in joined))
        self.assertTrue(any('exposure_time_absolute=80' in s for s in joined))
        self.assertTrue(any('white_balance_automatic=0' in s for s in joined))
        self.assertTrue(any('white_balance_temperature=4600' in s for s in joined))
        self.assertTrue(any('gain=12' in s for s in joined))

    def test_lock_on_start_rising_steady_falling_and_exposure_cap(self):
        calls = []
        logs = []

        def fake_runner(cmd, **_kwargs):
            calls.append(cmd)
            if '--get-ctrl' in cmd:
                ctrl = cmd[cmd.index('--get-ctrl') + 1]
                if ctrl == 'exposure_time_absolute':
                    return FakeCompletedProcess(0, 'exposure_time_absolute: 150\n', '')
                if ctrl == 'white_balance_temperature':
                    return FakeCompletedProcess(0, 'white_balance_temperature: 4800\n', '')
                if ctrl == 'gain':
                    return FakeCompletedProcess(0, 'gain: 18\n', '')
            return FakeCompletedProcess(0, '', '')

        policy = camera_mod.CameraV4L2Policy(
            device='/dev/video0',
            ae_awb_mode='lock_on_start',
            exposure_time_absolute=80,  # 8ms cap -> min(150, 80) = 80
            runner=fake_runner,
            logger=logs.append,
        )
        policy.apply_startup()
        self.assertFalse(policy.locked)

        calls.clear()
        action = policy.on_start_cmd(True)
        self.assertEqual(action, 'locked')
        self.assertTrue(policy.locked)
        self.assertEqual(policy.locked_values.get('exposure_time_absolute'), 80)
        self.assertEqual(policy.locked_values.get('white_balance_temperature'), 4800)
        self.assertEqual(policy.locked_values.get('gain'), 18)

        # Steady-state True must be O(1) noop with ZERO subprocess calls
        calls.clear()
        self.assertEqual(policy.on_start_cmd(True), 'noop')
        self.assertEqual(len(calls), 0)

        # Falling edge unlocks back to auto
        action = policy.on_start_cmd(False)
        self.assertEqual(action, 'unlocked')
        self.assertFalse(policy.locked)
        joined = [' '.join(c) for c in calls]
        self.assertTrue(any('auto_exposure=3' in s for s in joined))
        self.assertTrue(any('white_balance_automatic=1' in s for s in joined))
        self.assertTrue(any('[DEC] camera v4l2 lock_on_start' in m for m in logs))

    def test_missing_v4l2_ctl_and_timeout_fail_gracefully(self):
        logs = []

        def missing_runner(_cmd, **_kwargs):
            raise FileNotFoundError('v4l2-ctl not installed')

        p1 = camera_mod.CameraV4L2Policy(
            ae_awb_mode='lock_on_start', runner=missing_runner, logger=logs.append
        )
        p1.apply_startup()
        self.assertFalse(p1.v4l2_available)
        self.assertEqual(p1.on_start_cmd(True), 'locked')
        self.assertFalse(p1.locked)
        self.assertTrue(any('v4l2-ctl_not_found' in m for m in logs))

        def timeout_runner(_cmd, **_kwargs):
            raise subprocess.TimeoutExpired(cmd='v4l2-ctl', timeout=0.5)

        p2 = camera_mod.CameraV4L2Policy(
            ae_awb_mode='manual_fixed', exposure_time_absolute=80, runner=timeout_runner
        )
        p2.apply_startup()
        self.assertFalse(p2.locked)


class TestBPUYoloV8PostProcessAndTracker(unittest.TestCase):
    def test_class_list_files_and_dynamic_validation(self):
        cls8 = yolo_mod.load_class_list(str(WS_ROOT / 'src' / 'yolov8' / 'mod1' / 'out.list'))
        cls11 = yolo_mod.load_class_list(
            str(WS_ROOT / 'src' / 'yolov8' / 'mod1' / 'out_finals11.list')
        )
        self.assertEqual(cls8, ['1', '2', '3', '4', '5', '6', '7', '8'])
        self.assertEqual(
            cls11,
            ['1', '2', '3', '4', '5', '6', '7', '8', 'red_cone', 'red_no_entry', 'yellow_stop_line'],
        )

        # Matching 8-class and 11-class models (NHWC & NCHW) pass
        m8 = FakeBPUModel([(1, 80, 80, 8), (1, 40, 40, 8), (1, 20, 20, 8),
                           (1, 80, 80, 64), (1, 40, 40, 64), (1, 20, 20, 64)])
        m11 = FakeBPUModel([(1, 11, 80, 80), (1, 11, 40, 40), (1, 11, 20, 20),
                            (1, 64, 80, 80), (1, 64, 40, 40), (1, 64, 20, 20)])
        self.assertEqual(yolo_mod.validate_model_classes(m8, cls8), 8)
        self.assertEqual(yolo_mod.validate_model_classes(m11, cls11), 11)

        # Mismatched class count raises RuntimeError
        with self.assertRaises(RuntimeError):
            yolo_mod.validate_model_classes(m8, cls11)
        with self.assertRaises(RuntimeError):
            yolo_mod.validate_model_classes(m11, cls8)

    def test_decode_pre_sigmoid_filter_and_class_conf_overrides(self):
        cls11 = yolo_mod.load_class_list(
            str(WS_ROOT / 'src' / 'yolov8' / 'mod1' / 'out_finals11.list')
        )
        grids = _make_grids()
        dfl_w = np.arange(16, dtype=np.float32)

        # logit(0.18) ≈ -1.516, logit(0.22) ≈ -1.266, logit(0.85) ≈ +1.735
        injections = [
            # red_no_entry (idx 9) with prob ~0.18 -> should PASS when override=0.15
            {'stride_idx': 1, 'gy': 20, 'gx': 20, 'cls_id': 9, 'logit': -1.516, 'dfl_dist': (2, 2, 2, 2)},
            # digit '1' (idx 0) with prob ~0.22 -> should FAIL when override=0.30
            {'stride_idx': 1, 'gy': 10, 'gx': 10, 'cls_id': 0, 'logit': -1.266, 'dfl_dist': (2, 2, 2, 2)},
            # red_cone (idx 8) with prob ~0.85 -> should PASS
            {'stride_idx': 2, 'gy': 10, 'gx': 10, 'cls_id': 8, 'logit': 1.735, 'dfl_dist': (1, 2, 1, 2)},
        ]
        outputs = _build_synthetic_outputs(11, injections, layout='NHWC')

        thresholds = yolo_mod.parse_class_conf_overrides(
            '{"red_no_entry": 0.15, "1": 0.30}', cls11, default_conf=0.25
        )
        self.assertAlmostEqual(float(thresholds[9]), 0.15, places=5)
        self.assertAlmostEqual(float(thresholds[0]), 0.30, places=5)
        self.assertAlmostEqual(float(thresholds[8]), 0.25, places=5)

        all_boxes, all_scores, all_cls_ids, ok, cls_cnt, reg_cnt = yolo_mod.decode_yolov8_outputs(
            outputs=outputs,
            grids=grids,
            dfl_weights=dfl_w,
            num_classes=11,
            conf_thresh=0.25,
            pad_w=0.0,
            pad_h=80.0,
            scale_r=1.0,
            orig_w=640,
            orig_h=480,
            class_conf_thresholds=thresholds,
        )
        self.assertTrue(ok)
        self.assertEqual((cls_cnt, reg_cnt), (3, 3))

        dets = yolo_mod.run_nms_filter(
            all_boxes=all_boxes,
            all_scores=all_scores,
            all_cls_ids=all_cls_ids,
            classes=cls11,
            conf_thresh=0.25,
            iou_thresh=0.55,
            top_k=20,
            min_box_size=6,
            orig_w=640,
            orig_h=480,
            class_agnostic_nms=False,
            class_conf_thresholds=thresholds,
        )
        det_types = {d['type'] for d in dets}
        self.assertEqual(det_types, {'red_no_entry', 'red_cone'})

    def test_decode_per_channel_scale_nchw_and_nhwc(self):
        grids = _make_grids()
        dfl_w = np.arange(16, dtype=np.float32)
        injections = [
            {'stride_idx': 1, 'gy': 20, 'gx': 20, 'cls_id': 3, 'logit': 5.0,
             'dfl_dist': (2, 2, 2, 2)}
        ]

        for layout in ('NCHW', 'NHWC'):
            with self.subTest(layout=layout):
                outputs = _build_synthetic_outputs(8, injections, layout=layout)
                for idx, out in enumerate(outputs):
                    channels = 8 if idx % 2 == 0 else 64
                    out.properties.scale_data = [2.0] * channels

                boxes, scores, cls_ids, ok, cls_cnt, reg_cnt = yolo_mod.decode_yolov8_outputs(
                    outputs=outputs,
                    grids=grids,
                    dfl_weights=dfl_w,
                    num_classes=8,
                    conf_thresh=0.25,
                    pad_w=0.0,
                    pad_h=80.0,
                    scale_r=1.0,
                    orig_w=640,
                    orig_h=480,
                )
                self.assertTrue(ok)
                self.assertEqual((cls_cnt, reg_cnt), (3, 3))
                self.assertEqual(len(boxes), 1)
                self.assertEqual(int(np.asarray(cls_ids[0]).item()), 3)
                self.assertGreater(float(np.asarray(scores[0]).item()), 0.25)

    def test_per_class_nms_keeps_adjacent_different_classes_and_suppresses_same_class(self):
        cls11 = yolo_mod.load_class_list(
            str(WS_ROOT / 'src' / 'yolov8' / 'mod1' / 'out_finals11.list')
        )
        # Box 0 & Box 1: heavily overlapping DIFFERENT classes ('3' and '7', IoU ~0.82)
        # Box 2: duplicate of Box 0 SAME class ('3', IoU ~0.82) -> MUST be suppressed!
        boxes = [np.array([
            [100.0, 200.0, 50.0, 50.0],  # cls '3' (id 2), score 0.90
            [104.0, 202.0, 50.0, 50.0],  # cls '7' (id 6), score 0.85
            [103.0, 201.0, 50.0, 50.0],  # cls '3' (id 2), score 0.70 (dup)
        ], dtype=np.float32)]
        scores = [np.array([0.90, 0.85, 0.70], dtype=np.float32)]
        cls_ids = [np.array([2, 6, 2], dtype=np.int64)]

        # Default per-class NMS keeps BOTH '3' and '7', suppresses duplicate '3'
        dets_per_cls = yolo_mod.run_nms_filter(
            boxes, scores, cls_ids, cls11, 0.25, 0.45, 20, 6, 640, 480,
            class_agnostic_nms=False,
        )
        self.assertEqual(len(dets_per_cls), 2)
        self.assertEqual({d['type'] for d in dets_per_cls}, {'3', '7'})
        # Verify unshifted coordinates are preserved
        for d in dets_per_cls:
            self.assertLess(d['bx'], 200.0)
            self.assertLess(d['by'], 300.0)

        # Class-agnostic NMS suppresses '7' due to overlap with '3'
        dets_agnostic = yolo_mod.run_nms_filter(
            boxes, scores, cls_ids, cls11, 0.25, 0.45, 20, 6, 640, 480,
            class_agnostic_nms=True,
        )
        self.assertEqual(len(dets_agnostic), 1)
        self.assertEqual(dets_agnostic[0]['type'], '3')

    def test_simple_box_tracker_persistence_class_isolation_and_expiry(self):
        tracker = yolo_mod.SimpleBoxTracker(
            iou_thresh=0.30, max_center_dist_px=60.0, max_missed_frames=2
        )

        def mk(cls_name, bx, by, bw=40.0, bh=40.0):
            return {
                'track_id': 0,
                'type': cls_name,
                'confidence': 0.9,
                'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
                'cx': bx + bw / 2.0, 'cy': by + bh / 2.0,
            }

        # Frame 1: two different classes at identical location -> distinct track IDs
        f1 = tracker.update([mk('red_cone', 100.0, 200.0), mk('red_no_entry', 100.0, 200.0)])
        id_cone = [d['track_id'] for d in f1 if d['type'] == 'red_cone'][0]
        id_ne = [d['track_id'] for d in f1 if d['type'] == 'red_no_entry'][0]
        self.assertNotEqual(id_cone, id_ne)

        # Frame 2: slight motion (IoU match) preserves IDs and increments hits
        f2 = tracker.update([mk('red_cone', 108.0, 205.0), mk('red_no_entry', 104.0, 202.0)])
        self.assertEqual([d['track_id'] for d in f2 if d['type'] == 'red_cone'][0], id_cone)
        self.assertEqual([d['track_hits'] for d in f2 if d['type'] == 'red_cone'][0], 2)
        self.assertEqual([d['track_id'] for d in f2 if d['type'] == 'red_no_entry'][0], id_ne)

        # Frame 3: fast jump (zero IoU, center dist=45px <= 60px) -> centroid fallback keeps ID
        f3 = tracker.update([mk('red_cone', 153.0, 205.0)])
        self.assertEqual(f3[0]['track_id'], id_cone)
        self.assertEqual(f3[0]['track_hits'], 3)

        # Frames 4, 5, 6: empty -> expires after max_missed_frames=2 (missed=1,2,3>2)
        tracker.update([])
        tracker.update([])
        tracker.update([])

        # Frame 7: reappearance gets a fresh track_id
        f7 = tracker.update([mk('red_cone', 153.0, 205.0)])
        self.assertGreater(f7[0]['track_id'], max(id_cone, id_ne))
        self.assertEqual(f7[0]['track_hits'], 1)

    def test_decode_nchw_with_per_channel_scale_data(self):
        """Regression test: NCHW outputs with per-channel scale_data must not broadcast-crash."""
        grids = _make_grids()
        dfl_w = np.arange(16, dtype=np.float32)
        injections = [
            {'stride_idx': 1, 'gy': 20, 'gx': 20, 'cls_id': 4, 'logit': 4.0, 'dfl_dist': (2, 2, 2, 2)},
        ]
        outputs = _build_synthetic_outputs(8, injections, layout='NCHW')
        for out in outputs:
            ch = out.buffer.shape[1]
            out.properties.scale_data = [0.5] * ch
        all_boxes, all_scores, all_cls_ids, ok, cls_cnt, reg_cnt = yolo_mod.decode_yolov8_outputs(
            outputs=outputs,
            grids=grids,
            dfl_weights=dfl_w,
            num_classes=8,
            conf_thresh=0.25,
            pad_w=0.0,
            pad_h=80.0,
            scale_r=1.0,
            orig_w=640,
            orig_h=480,
        )
        self.assertTrue(ok)
        self.assertEqual((cls_cnt, reg_cnt), (3, 3))
        self.assertEqual(sum(len(s) for s in all_scores), 1)
        self.assertEqual(int(all_cls_ids[0][0]), 4)

    def test_image_qos_best_effort_prevents_fastdds_stall(self):
        cam_src = (WS_ROOT / 'src' / 'camera' / 'camera.py').read_text(encoding='utf-8')
        yolo_src = (WS_ROOT / 'src' / 'yolov8' / 'bpu_yolov8_node.py').read_text(encoding='utf-8')
        panel_src = (WS_ROOT / 'tools' / 'panel' / 'server.py').read_text(encoding='utf-8')
        lf_src = (
            WS_ROOT / 'src' / 'line_follow_ws' / 'src' / 'line_follower'
            / 'src' / 'line_follower_node.cpp'
        ).read_text(encoding='utf-8')
        self.assertIn("declare_parameter('qos_depth', 4)", cam_src)
        self.assertIn('ReliabilityPolicy.BEST_EFFORT', yolo_src)
        self.assertIn('ReliabilityPolicy.BEST_EFFORT', panel_src)
        self.assertIn('rmw_qos_profile_sensor_data', lf_src)
        self.assertIn('debug_image_max_fps', lf_src)

    def test_windowed_perf_counters_async_v4l2_and_start_toggle_eof(self):
        cam_src = (WS_ROOT / 'src' / 'camera' / 'camera.py').read_text(encoding='utf-8')
        yolo_src = (WS_ROOT / 'src' / 'yolov8' / 'bpu_yolov8_node.py').read_text(encoding='utf-8')
        st_src = (
            WS_ROOT / 'src' / 'line_follow_ws' / 'src' / 'line_follower'
            / 'src' / 'start_toggle_node.cpp'
        ).read_text(encoding='utf-8')
        self.assertIn('camera_v4l2_worker', cam_src)
        self.assertIn('fail_delta = self.fail_count - self._last_stats_fails', cam_src)
        self.assertIn('dropped_delta = self.frame_slot.dropped_count - self._last_stats_dropped_count', yolo_src)
        self.assertIn('dropped_total=', yolo_src)
        self.assertIn('stdin_eof_', st_src)
        self.assertIn('rclcpp::shutdown()', st_src)

        tr = yolo_mod.StageLatencyTracker(window_size=10)
        rep = tr.format_report(proc_fps=30.0, rx_fps=30.0, dropped_frames=0, debug_pub_frames=12, dropped_total=18)
        self.assertIn('dropped=0 dropped_total=18', rep)


class TestSafetyWatchdogLogic(unittest.TestCase):
    def test_upstream_topics_exclude_detection_channel(self):
        self.assertEqual(safety_mod.UPSTREAM_TOPICS, ['/cmd_vel_follower', '/cmd_vel_stop'])
        safety_src = (WS_ROOT / 'tools' / 'safety_node.py').read_text(encoding='utf-8')
        self.assertNotIn('/cmd_vel_detection', safety_src)

    def test_idle_normal_and_upstream_silence(self):
        wd = safety_mod.SafetyWatchdogLogic(
            silence_timeout=0.30, mux_timeout_sec=0.35, mux_startup_grace_sec=3.0
        )
        # Idle before first message
        r0 = wd.evaluate(1.0)
        self.assertFalse(r0['should_brake'])
        self.assertEqual(r0['reason'], 'idle')

        # Normal streaming
        wd.on_upstream('/cmd_vel_follower', 2.0)
        wd.on_mux_heartbeat(2.0)
        r1 = wd.evaluate(2.1)
        self.assertFalse(r1['should_brake'])

        # Upstream goes silent while mux stays alive -> triggers braking
        wd.on_mux_heartbeat(2.4)
        r2 = wd.evaluate(2.45)
        self.assertTrue(r2['should_brake'])
        self.assertEqual(r2['transition'], 'deadman_on')
        self.assertIn('upstream_silence', r2['reason'])

        # Upstream resumes -> releases braking
        wd.on_upstream('/cmd_vel_follower', 2.50)
        wd.on_mux_heartbeat(2.50)
        r3 = wd.evaluate(2.55)
        self.assertFalse(r3['should_brake'])
        self.assertEqual(r3['transition'], 'deadman_off')

    def test_default_silence_timeout_tolerates_startup_jitter(self):
        wd = safety_mod.SafetyWatchdogLogic(mux_timeout_sec=0.35)
        self.assertEqual(wd.silence_timeout, 0.5)

        wd.on_upstream('/cmd_vel_stop', 10.0)
        wd.on_mux_heartbeat(10.0)
        self.assertFalse(wd.evaluate(10.32)['should_brake'])

        wd.on_mux_heartbeat(10.5)
        r = wd.evaluate(10.55)
        self.assertTrue(r['should_brake'])
        self.assertEqual(r['transition'], 'deadman_on')
        self.assertIn('upstream_silence', r['reason'])

    def test_cmd_mux_crash_while_follower_keeps_publishing_triggers_brake(self):
        """
        Regression test for the critical STM32 latch blind spot:
        line_follower_node keeps publishing /cmd_vel_follower at 20Hz,
        but cmd_mux_node crashes (stops /cmd_mux/heartbeat).
        safety_node MUST brake within mux_timeout_sec and NOT self-lock on its own zeros.
        """
        wd = safety_mod.SafetyWatchdogLogic(
            silence_timeout=0.30, mux_timeout_sec=0.35, mux_startup_grace_sec=3.0
        )
        wd.on_upstream('/cmd_vel_follower', 10.0)
        wd.on_mux_heartbeat(10.0)
        wd.on_cmd_vel_echo(10.0, linear_x=0.5, angular_z=0.1)
        self.assertFalse(wd.evaluate(10.1)['should_brake'])

        # Follower keeps publishing, mux heartbeat stops after t=10.0
        wd.on_upstream('/cmd_vel_follower', 10.2)
        wd.on_upstream('/cmd_vel_follower', 10.3)
        wd.on_upstream('/cmd_vel_follower', 10.4)
        r_crash = wd.evaluate(10.40)  # mux silence = 0.40 > 0.35
        self.assertTrue(r_crash['should_brake'])
        self.assertEqual(r_crash['transition'], 'deadman_on')
        self.assertEqual(r_crash['reason'], 'mux_heartbeat_timeout')

        # Receiving our own zero /cmd_vel echo must NOT clear braking
        wd.on_cmd_vel_echo(10.45, linear_x=0.0, angular_z=0.0)
        wd.on_upstream('/cmd_vel_follower', 10.45)
        r_still = wd.evaluate(10.46)
        self.assertTrue(r_still['should_brake'])

        # Once cmd_mux_node recovers and sends heartbeat, deadman releases
        wd.on_mux_heartbeat(10.50)
        r_rec = wd.evaluate(10.52)
        self.assertFalse(r_rec['should_brake'])
        self.assertEqual(r_rec['transition'], 'deadman_off')

    def test_mux_heartbeat_never_seen_after_grace_triggers_brake(self):
        wd = safety_mod.SafetyWatchdogLogic(
            silence_timeout=0.30, mux_timeout_sec=0.35, mux_startup_grace_sec=2.0
        )
        wd.on_upstream('/cmd_vel_follower', 1.0)
        self.assertFalse(wd.evaluate(1.2)['should_brake'])
        wd.on_upstream('/cmd_vel_follower', 3.1)
        r = wd.evaluate(3.2)  # 2.2s since first upstream, never saw mux hb
        self.assertTrue(r['should_brake'])
        self.assertEqual(r['reason'], 'mux_heartbeat_missing')


class TestConfigAndLaunchHygiene(unittest.TestCase):
    def test_ekf_yaml_uses_madgwick_imu_data(self):
        ekf_text = (
            WS_ROOT / 'src' / 'car_ws' / 'src' / 'turn_on_wheeltec_robot' / 'config' / 'ekf.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('imu0: /imu/data', ekf_text)
        self.assertNotIn('imu0: /imu/data_raw', ekf_text)

    def test_base_serial_launch_exposes_odom_scale_args(self):
        launch_text = (
            WS_ROOT / 'src' / 'car_ws' / 'src' / 'turn_on_wheeltec_robot'
            / 'launch' / 'base_serial.launch.py'
        ).read_text(encoding='utf-8')
        for arg in ('odom_x_scale', 'odom_y_scale', 'odom_z_scale_positive', 'odom_z_scale_negative'):
            self.assertIn(f"'{arg}'", launch_text)
        self.assertIn('DeclareLaunchArgument', launch_text)
        self.assertIn('ParameterValue', launch_text)

    def test_line_follower_with_detection_launch_no_detection_turn_controller(self):
        launch_text = (
            WS_ROOT / 'src' / 'line_follow_ws' / 'src' / 'line_follower'
            / 'launch' / 'line_follower_with_detection.launch.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('"high_timeout"', launch_text)
        self.assertNotIn('"stop_timeout"', launch_text)
        self.assertNotIn('detection_turn_controller', launch_text)
        self.assertIn('line_follower_node', launch_text)
        self.assertIn('cmd_mux_node', launch_text)

    def test_cmd_mux_node_two_channel_watchdog_and_heartbeat(self):
        cpp_text = (
            WS_ROOT / 'src' / 'line_follow_ws' / 'src' / 'line_follower'
            / 'src' / 'cmd_mux_node.cpp'
        ).read_text(encoding='utf-8')
        for token in (
            'watchdog_rate_hz',
            '/cmd_mux/heartbeat',
            'create_wall_timer',
        ):
            self.assertIn(token, cpp_text)
        for legacy_tok in ('enable_detection_override', 'session_cooldown', '/cmd_vel_detection'):
            self.assertNotIn(legacy_tok, cpp_text)

    def test_start_sh_b2_params_and_two_node_health_check(self):
        start_text = (WS_ROOT / 'tools' / 'start.sh').read_text(encoding='utf-8')
        self.assertIn('PARAMS="finals_fork"', start_text)
        self.assertIn('可选: finals_fork / finals_roundabout', start_text)
        self.assertIn("python3 -u camera.py --ros-args --params-file '$PARAMS_FILE'", start_text)
        self.assertIn("python3 -u bpu_yolov8_node.py --ros-args --params-file '$PARAMS_FILE'", start_text)
        self.assertIn('for exe in line_follower_node cmd_mux_node; do', start_text)
        self.assertIn('-lt 2', start_text)
        self.assertIn('$cnt/2', start_text)
        self.assertIn('Line follower (finals) subscribing', start_text)
        self.assertNotIn('Line follower subscribing', start_text)
        self.assertNotIn('detection_turn_controller', start_text)

        for sim_sh in ('run_sim.sh', 'verify.sh'):
            sim_text = (WS_ROOT / 'tools' / 'sim' / sim_sh).read_text(encoding='utf-8')
            self.assertIn('finals_fork', sim_text)
            self.assertNotIn('detection_turn_controller', sim_text)
        verify_text = (WS_ROOT / 'tools' / 'sim' / 'verify.sh').read_text(encoding='utf-8')
        self.assertIn('corridor_kp_y', verify_text)
        self.assertIn('yellow_stop_min_time', verify_text)
        self.assertIn(r'count=(\d+)', verify_text)
        self.assertNotIn('yellow_stop_duration', verify_text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
