"""ctypes wrapper around the real-car pure C++17 brain DLL (tools/sim/build/sim_brain.dll).

Loads parameters directly from tools/config/finals_fork.yaml or finals_roundabout.yaml
(under line_follower_node.ros__parameters), applies optional overrides, and steps the exact
7 pure C++17 modules identically to LineFollowerNode.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from .build_brain_dll import DLL_PATH, build_dll
except ImportError:
    from build_brain_dll import DLL_PATH, build_dll


# SemanticObjectType enum values matching cpp_brain_bridge.cpp
OBJ_UNKNOWN = 0
OBJ_RED_CONE = 1
OBJ_RED_NO_ENTRY = 2
OBJ_BONUS_DIGIT = 3
OBJ_YELLOW_STOP_LINE = 4


class BrainDetectionC(ctypes.Structure):
    _fields_ = [
        ("obj_type", ctypes.c_int32),
        ("digit", ctypes.c_int32),
        ("confidence", ctypes.c_double),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("track_id", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class BrainCvRedBoxC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("fill_ratio", ctypes.c_double),
        ("is_no_entry", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class BrainStepInput(ctypes.Structure):
    _fields_ = [
        ("image_width", ctypes.c_int32),
        ("image_height", ctypes.c_int32),
        ("lane_valid", ctypes.c_int32),
        ("near_valid", ctypes.c_int32),
        ("far_valid", ctypes.c_int32),
        ("has_lateral_error", ctypes.c_int32),
        ("lateral_error", ctypes.c_double),
        ("near_error", ctypes.c_double),
        ("far_error", ctypes.c_double),
        ("heading_error", ctypes.c_double),
        ("curvature_proxy", ctypes.c_double),
        ("confidence", ctypes.c_double),
        ("near_width_ratio", ctypes.c_double),
        ("far_width_ratio", ctypes.c_double),
        ("lane_width_ratio", ctypes.c_double),
        ("has_left_edge", ctypes.c_int32),
        ("has_right_edge", ctypes.c_int32),
        ("edges_are_px", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("left_edge", ctypes.c_double),
        ("right_edge", ctypes.c_double),
        ("fork_detected", ctypes.c_int32),
        ("use_external_reverse_curve", ctypes.c_int32),
        ("reverse_curve_active", ctypes.c_int32),
        ("_pad1", ctypes.c_int32),
        ("left_branch_error", ctypes.c_double),
        ("right_branch_error", ctypes.c_double),
        ("yellow_detected", ctypes.c_int32),
        ("yellow_is_geometric", ctypes.c_int32),
        ("yellow_dead_end", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
        ("yellow_bottom_y", ctypes.c_double),
        ("yellow_center_x", ctypes.c_double),
        ("yellow_width_ratio", ctypes.c_double),
        ("has_odom", ctypes.c_int32),
        ("_pad3", ctypes.c_int32),
        ("odom_x", ctypes.c_double),
        ("odom_y", ctypes.c_double),
        ("odom_yaw", ctypes.c_double),
        ("odom_v", ctypes.c_double),
        ("odom_w", ctypes.c_double),
        ("voltage", ctypes.c_double),
        ("timestamp_sec", ctypes.c_double),
        ("detections", ctypes.POINTER(BrainDetectionC)),
        ("num_detections", ctypes.c_int32),
        ("_pad4", ctypes.c_int32),
        ("cv_red_boxes", ctypes.POINTER(BrainCvRedBoxC)),
        ("num_cv_red_boxes", ctypes.c_int32),
        ("_pad5", ctypes.c_int32),
    ]


class BrainStepOutput(ctypes.Structure):
    _fields_ = [
        ("v_cmd", ctypes.c_double),
        ("w_cmd", ctypes.c_double),
        ("state", ctypes.c_int32),
        ("publish_stop", ctypes.c_int32),
        ("corridor_left_m", ctypes.c_double),
        ("corridor_right_m", ctypes.c_double),
        ("corridor_target_y_m", ctypes.c_double),
        ("lane_left_m", ctypes.c_double),
        ("lane_right_m", ctypes.c_double),
        ("speed_cap_mps", ctypes.c_double),
        ("active_digit", ctypes.c_int32),
        ("digit_target_active", ctypes.c_int32),
        ("obstacle_count", ctypes.c_int32),
        ("near_obstacle_count", ctypes.c_int32),
        ("far_obstacle_count", ctypes.c_int32),
        ("has_blocking_obstacle", ctypes.c_int32),
        ("preferred_branch", ctypes.c_int32),
        ("cbf_active", ctypes.c_int32),
        ("cbf_min_clearance_m", ctypes.c_double),
        ("finish_armed", ctypes.c_int32),
        ("permanent_stop", ctypes.c_int32),
        ("yellow_stop_active", ctypes.c_int32),
        ("yellow_brake_active", ctypes.c_int32),
        ("yellow_count", ctypes.c_int32),
        ("yellow_dist_valid", ctypes.c_int32),
        ("yellow_is_finish", ctypes.c_int32),
        ("yellow_rearmed", ctypes.c_int32),
        ("yellow_d_bumper_m", ctypes.c_double),
        ("total_arc_m", ctypes.c_double),
        ("friction_scale", ctypes.c_double),
        ("effective_brake_decel", ctypes.c_double),
        ("corridor_left_px", ctypes.c_int32),
        ("corridor_right_px", ctypes.c_int32),
        ("corridor_ref_px", ctypes.c_int32),
        ("corridor_row_y", ctypes.c_int32),
        ("state_name", ctypes.c_char * 32),
        ("active_mode_tag", ctypes.c_char * 32),
        ("yellow_phase", ctypes.c_char * 32),
        ("cbf_limiting_source", ctypes.c_char * 32),
    ]


_DLL_CACHE: Optional[ctypes.CDLL] = None
_DLL_DIR_HANDLES: List[Any] = []


def _load_brain_dll(auto_build: bool = True) -> ctypes.CDLL:
    global _DLL_CACHE
    if _DLL_CACHE is not None:
        return _DLL_CACHE

    if auto_build:
        dll_path = build_dll(force=False, verbose=False)
    else:
        dll_path = DLL_PATH
        if not dll_path.is_file():
            raise FileNotFoundError(f"Brain DLL not found: {dll_path}")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        # Ensure MinGW runtime directory (e.g. libmcfgthread / libwinpthread) is discoverable
        gxx = shutil.which("g++")
        if gxx:
            mingw_bin = str(Path(gxx).resolve().parent)
            if os.path.isdir(mingw_bin):
                _DLL_DIR_HANDLES.append(os.add_dll_directory(mingw_bin))
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_path.parent)))

    lib = ctypes.CDLL(str(dll_path))

    lib.brain_create.argtypes = []
    lib.brain_create.restype = ctypes.c_void_p

    lib.brain_destroy.argtypes = [ctypes.c_void_p]
    lib.brain_destroy.restype = None

    lib.brain_reset.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.brain_reset.restype = None

    lib.brain_set_param_double.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.brain_set_param_double.restype = ctypes.c_int32

    lib.brain_set_param_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.brain_set_param_int.restype = ctypes.c_int32

    lib.brain_set_param_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.brain_set_param_bool.restype = ctypes.c_int32

    lib.brain_set_param_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.brain_set_param_string.restype = ctypes.c_int32

    lib.brain_set_homography.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    ]
    lib.brain_set_homography.restype = ctypes.c_int32

    lib.brain_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(BrainStepInput),
        ctypes.POINTER(BrainStepOutput),
    ]
    lib.brain_step.restype = ctypes.c_int32

    lib.brain_get_preferred_branch.argtypes = [ctypes.c_void_p]
    lib.brain_get_preferred_branch.restype = ctypes.c_int32

    lib.brain_pixel_to_ground.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.brain_pixel_to_ground.restype = ctypes.c_int32

    _DLL_CACHE = lib
    return lib


def classify_detection_label(label: str) -> Tuple[int, int]:
    """Classify detection label identically to LineFollowerNode::classify_detection_label."""
    if not label:
        return OBJ_UNKNOWN, 0
    lower = str(label).strip().lower()
    if lower in ("red_cone", "cone"):
        return OBJ_RED_CONE, 0
    if lower in ("red_no_entry", "no_entry", "forbid", "forbidden"):
        return OBJ_RED_NO_ENTRY, 0
    if lower in ("yellow_stop_line", "yellow_line", "stop_line"):
        return OBJ_YELLOW_STOP_LINE, 0

    num_part = lower
    if num_part.startswith("digit_"):
        num_part = num_part[6:]
    elif len(num_part) >= 2 and num_part[0] == "d":
        num_part = num_part[1:]
    if len(num_part) == 1 and "1" <= num_part <= "8":
        return OBJ_BONUS_DIGIT, int(num_part)
    return OBJ_UNKNOWN, 0


def load_line_follower_params(config_yaml_path: str | Path) -> Dict[str, Any]:
    """Parse line_follower_node.ros__parameters from a ROS2 YAML config file."""
    path = Path(config_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    if "line_follower_node" in raw and isinstance(raw["line_follower_node"], dict):
        node_cfg = raw["line_follower_node"]
        if "ros__parameters" in node_cfg and isinstance(node_cfg["ros__parameters"], dict):
            return dict(node_cfg["ros__parameters"])
        return dict(node_cfg)
    if "ros__parameters" in raw and isinstance(raw["ros__parameters"], dict):
        return dict(raw["ros__parameters"])
    return dict(raw)


def _extract_bbox_xywh(item: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(item, dict):
        if "bbox" in item and item["bbox"] is not None:
            b = item["bbox"]
            if len(b) == 4:
                return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        if "xyxy" in item and item["xyxy"] is not None:
            b = item["xyxy"]
            if len(b) == 4:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)
        if all(k in item for k in ("x", "y", "w", "h")):
            return float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"])
        if all(k in item for k in ("x", "y", "width", "height")):
            return (
                float(item["x"]),
                float(item["y"]),
                float(item["width"]),
                float(item["height"]),
            )
        if all(k in item for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = float(item["x1"]), float(item["y1"])
            x2, y2 = float(item["x2"]), float(item["y2"])
            return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)
        return None
    if isinstance(item, (tuple, list)) and len(item) >= 4:
        return float(item[0]), float(item[1]), float(item[2]), float(item[3])
    return None


class BrainBridge:
    """Python ctypes interface to the real-car pure C++17 brain DLL."""

    def __init__(
        self,
        config_yaml_path: str | Path,
        param_overrides: Optional[Dict[str, Any]] = None,
        auto_build: bool = True,
    ) -> None:
        self._lib = _load_brain_dll(auto_build=auto_build)
        self._handle: Optional[int] = self._lib.brain_create()
        if not self._handle:
            raise RuntimeError("brain_create() returned NULL")

        self.config_yaml_path = str(config_yaml_path)
        self.params: Dict[str, Any] = load_line_follower_params(config_yaml_path)
        if param_overrides:
            self.params.update(param_overrides)

        self.update_params(self.params)
        self.reset(0.0)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.brain_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "BrainBridge":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _require_handle(self) -> int:
        if not self._handle:
            raise RuntimeError("BrainBridge handle is closed")
        return self._handle

    def set_param(self, name: str, value: Any) -> int:
        handle = self._require_handle()
        if name == "homography_matrix":
            if isinstance(value, (list, tuple)) and len(value) == 9:
                arr = (ctypes.c_double * 9)(*[float(v) for v in value])
                enable = 1 if bool(self.params.get("use_homography", False)) else 0
                self.params[name] = [float(v) for v in value]
                return int(self._lib.brain_set_homography(handle, arr, enable))
            return 0

        self.params[name] = value
        name_b = name.encode("utf-8")
        if isinstance(value, bool):
            rc = int(self._lib.brain_set_param_bool(handle, name_b, 1 if value else 0))
            if name == "use_homography":
                h_mat = self.params.get("homography_matrix")
                if isinstance(h_mat, (list, tuple)) and len(h_mat) == 9:
                    arr = (ctypes.c_double * 9)(*[float(v) for v in h_mat])
                    self._lib.brain_set_homography(handle, arr, 1 if value else 0)
            return rc
        if isinstance(value, int):
            return int(self._lib.brain_set_param_int(handle, name_b, int(value)))
        if isinstance(value, float):
            return int(self._lib.brain_set_param_double(handle, name_b, float(value)))
        if isinstance(value, str):
            return int(self._lib.brain_set_param_string(handle, name_b, value.encode("utf-8")))
        return 0

    def update_params(self, overrides: Dict[str, Any]) -> None:
        # Apply homography_matrix after use_homography so both stay consistent
        deferred_h = None
        for k, v in overrides.items():
            if k == "homography_matrix":
                deferred_h = v
                continue
            self.set_param(k, v)
        if deferred_h is not None:
            self.set_param("homography_matrix", deferred_h)

    def reset(self, now_sec: float = 0.0) -> None:
        handle = self._require_handle()
        self._lib.brain_reset(handle, float(now_sec))

    @property
    def preferred_branch(self) -> int:
        handle = self._require_handle()
        return int(self._lib.brain_get_preferred_branch(handle))

    def pixel_to_ground(self, u: float, v: float) -> Tuple[bool, float, float]:
        handle = self._require_handle()
        d_out = ctypes.c_double(0.0)
        y_out = ctypes.c_double(0.0)
        ok = self._lib.brain_pixel_to_ground(
            handle, float(u), float(v), ctypes.byref(d_out), ctypes.byref(y_out)
        )
        return bool(ok), float(d_out.value), float(y_out.value)

    def _pack_detections(self, detections: Optional[Iterable[Any]]) -> Any:
        items: List[BrainDetectionC] = []
        if detections:
            for det in detections:
                if not isinstance(det, dict):
                    continue
                bbox = _extract_bbox_xywh(det)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                if w <= 0.0 or h <= 0.0:
                    continue

                label = det.get("class_name") or det.get("label") or det.get("type") or ""
                if isinstance(label, int):
                    obj_type = int(label)
                    digit = int(det.get("digit", 0) or 0)
                else:
                    obj_type, digit = classify_detection_label(str(label))
                    if digit == 0 and det.get("digit"):
                        digit = int(det["digit"])
                        if 1 <= digit <= 8 and obj_type == OBJ_UNKNOWN:
                            obj_type = OBJ_BONUS_DIGIT
                if obj_type == OBJ_UNKNOWN:
                    continue

                conf = det.get("conf", det.get("confidence", 0.8))
                track_id = det.get("track_id", -1)
                items.append(
                    BrainDetectionC(
                        obj_type=obj_type,
                        digit=digit,
                        confidence=float(conf),
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        track_id=int(track_id) if track_id is not None else -1,
                        _pad=0,
                    )
                )
        if not items:
            return None, 0
        arr = (BrainDetectionC * len(items))(*items)
        return arr, len(items)

    def _pack_cv_red_boxes(self, cv_red_boxes: Optional[Iterable[Any]]) -> Any:
        items: List[BrainCvRedBoxC] = []
        if cv_red_boxes:
            for box in cv_red_boxes:
                bbox = _extract_bbox_xywh(box)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                if w <= 0.0 or h <= 0.0:
                    continue

                is_no_entry = 0
                confidence = 0.60
                fill_ratio = -1.0
                if isinstance(box, dict):
                    if "is_no_entry" in box:
                        is_no_entry = 1 if bool(box["is_no_entry"]) else 0
                    else:
                        cls_str = str(box.get("type") or box.get("label") or box.get("class_name") or "").lower()
                        if cls_str in ("no_entry", "red_no_entry", "forbid", "forbidden"):
                            is_no_entry = 1
                    if "confidence" in box or "conf" in box:
                        confidence = float(box.get("confidence", box.get("conf", 0.60)))
                    if "fill_ratio" in box and box["fill_ratio"] is not None:
                        fill_ratio = float(box["fill_ratio"])
                elif isinstance(box, (tuple, list)) and len(box) >= 5:
                    v4 = box[4]
                    if isinstance(v4, str):
                        is_no_entry = 1 if v4.lower() in ("no_entry", "red_no_entry") else 0
                    else:
                        is_no_entry = 1 if bool(v4) else 0

                items.append(
                    BrainCvRedBoxC(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        confidence=confidence,
                        fill_ratio=fill_ratio,
                        is_no_entry=is_no_entry,
                        _pad=0,
                    )
                )
        if not items:
            return None, 0
        arr = (BrainCvRedBoxC * len(items))(*items)
        return arr, len(items)

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Step the C++17 brain with a vision + odom observation dict."""
        handle = self._require_handle()

        img_w = int(obs.get("image_width", obs.get("frame_width", 640)))
        img_h = int(obs.get("image_height", obs.get("frame_height", 480)))
        lane_valid = bool(obs.get("lane_valid", False))

        near_valid = -1
        if "near_valid" in obs and obs["near_valid"] is not None:
            near_valid = 1 if bool(obs["near_valid"]) else 0

        far_valid = -1
        if "far_valid" in obs and obs["far_valid"] is not None:
            far_valid = 1 if bool(obs["far_valid"]) else 0

        has_lat_err = 0
        lat_err = 0.0
        if "lateral_error" in obs and obs["lateral_error"] is not None:
            has_lat_err = 1
            lat_err = float(obs["lateral_error"])
        elif "error" in obs and obs["error"] is not None:
            has_lat_err = 1
            lat_err = float(obs["error"])

        near_error = float(obs.get("near_error", 0.0))
        far_error = float(obs.get("far_error", 0.0))
        heading_error = float(obs.get("heading_error", obs.get("heading_proxy", 0.0)))
        curvature_proxy = float(obs["curvature_proxy"]) if obs.get("curvature_proxy") is not None else -1.0
        confidence = float(obs.get("confidence", 1.0 if lane_valid else 0.0))
        near_width_ratio = float(obs.get("near_width_ratio", 0.68 if lane_valid else 0.0))
        far_width_ratio = float(obs.get("far_width_ratio", 0.48 if lane_valid else 0.0))
        lane_width_ratio = (
            float(obs["lane_width_ratio"]) if obs.get("lane_width_ratio") is not None else -1.0
        )

        # Support either pixel edges (near_left_edge_px / near_right_edge_px) or norm [-1, 1] edges
        if "near_left_edge_px" in obs and "near_right_edge_px" in obs:
            edges_are_px = 1
            left_edge = float(obs["near_left_edge_px"])
            right_edge = float(obs["near_right_edge_px"])
        else:
            edges_are_px = 0
            left_edge = float(obs.get("left_edge_norm", -1.0))
            right_edge = float(obs.get("right_edge_norm", 1.0))

        has_left_edge = bool(obs.get("has_left_edge", obs.get("near_left_valid", False)))
        has_right_edge = bool(obs.get("has_right_edge", obs.get("near_right_valid", False)))

        fork_detected = bool(obs.get("fork_detected", obs.get("has_fork_split", False)))
        left_branch_error = float(obs.get("left_branch_error", 0.0))
        right_branch_error = float(obs.get("right_branch_error", 0.0))

        use_ext_rev = 0
        ext_rev = 0
        if "reverse_curve_active" in obs and obs["reverse_curve_active"] is not None:
            use_ext_rev = 1
            ext_rev = 1 if bool(obs["reverse_curve_active"]) else 0

        yellow_detected = bool(obs.get("yellow_detected", False))
        yellow_bottom_y = float(
            obs.get("yellow_bottom_y_full", obs.get("yellow_bottom_y", 0.0))
        )
        yellow_center_x = float(
            obs.get("yellow_center_x_full", obs.get("yellow_center_x", -1.0))
        )
        yellow_width_ratio = float(obs.get("yellow_width_ratio", 0.0))
        yellow_dead_end = bool(
            obs.get("yellow_dead_end", obs.get("yellow_has_dead_end_ahead", False))
        )
        if "yellow_is_geometric" in obs and obs["yellow_is_geometric"] is not None:
            yellow_is_geometric = 1 if bool(obs["yellow_is_geometric"]) else 0
        else:
            yellow_is_geometric = 1 if (yellow_detected and yellow_bottom_y > 0.0) else 0

        # Odom fields (flat keys or nested obs["odom"] dict)
        odom_dict = obs.get("odom") if isinstance(obs.get("odom"), dict) else None
        has_odom = any(
            k in obs for k in ("odom_x", "odom_y", "odom_yaw", "odom_v", "odom_w")
        ) or (odom_dict is not None)
        if odom_dict is not None:
            odom_x = float(obs.get("odom_x", odom_dict.get("x", 0.0)))
            odom_y = float(obs.get("odom_y", odom_dict.get("y", 0.0)))
            odom_yaw = float(obs.get("odom_yaw", odom_dict.get("yaw", 0.0)))
            odom_v = float(obs.get("odom_v", odom_dict.get("v", odom_dict.get("linear_v", 0.0))))
            odom_w = float(obs.get("odom_w", odom_dict.get("w", odom_dict.get("angular_w", 0.0))))
        else:
            odom_x = float(obs.get("odom_x", 0.0))
            odom_y = float(obs.get("odom_y", 0.0))
            odom_yaw = float(obs.get("odom_yaw", 0.0))
            odom_v = float(obs.get("odom_v", 0.0))
            odom_w = float(obs.get("odom_w", 0.0))

        raw_voltage = obs.get("voltage", 0.0)
        voltage = float(raw_voltage) if (raw_voltage is not None and math.isfinite(float(raw_voltage))) else 0.0
        timestamp_sec = float(obs.get("timestamp_sec", obs.get("stamp_sec", obs.get("t", 0.0))))

        det_arr, num_dets = self._pack_detections(obs.get("detections"))
        cv_arr, num_cv = self._pack_cv_red_boxes(obs.get("cv_red_boxes"))

        step_in = BrainStepInput(
            image_width=img_w,
            image_height=img_h,
            lane_valid=1 if lane_valid else 0,
            near_valid=near_valid,
            far_valid=far_valid,
            has_lateral_error=has_lat_err,
            lateral_error=lat_err,
            near_error=near_error,
            far_error=far_error,
            heading_error=heading_error,
            curvature_proxy=curvature_proxy,
            confidence=confidence,
            near_width_ratio=near_width_ratio,
            far_width_ratio=far_width_ratio,
            lane_width_ratio=lane_width_ratio,
            has_left_edge=1 if has_left_edge else 0,
            has_right_edge=1 if has_right_edge else 0,
            edges_are_px=edges_are_px,
            _pad0=0,
            left_edge=left_edge,
            right_edge=right_edge,
            fork_detected=1 if fork_detected else 0,
            use_external_reverse_curve=use_ext_rev,
            reverse_curve_active=ext_rev,
            _pad1=0,
            left_branch_error=left_branch_error,
            right_branch_error=right_branch_error,
            yellow_detected=1 if yellow_detected else 0,
            yellow_is_geometric=yellow_is_geometric,
            yellow_dead_end=1 if yellow_dead_end else 0,
            _pad2=0,
            yellow_bottom_y=yellow_bottom_y,
            yellow_center_x=yellow_center_x,
            yellow_width_ratio=yellow_width_ratio,
            has_odom=1 if has_odom else 0,
            _pad3=0,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_yaw=odom_yaw,
            odom_v=odom_v,
            odom_w=odom_w,
            voltage=voltage,
            timestamp_sec=timestamp_sec,
            detections=det_arr,
            num_detections=num_dets,
            _pad4=0,
            cv_red_boxes=cv_arr,
            num_cv_red_boxes=num_cv,
            _pad5=0,
        )
        step_out = BrainStepOutput()
        rc = self._lib.brain_step(handle, ctypes.byref(step_in), ctypes.byref(step_out))
        if rc != 0:
            raise RuntimeError(f"brain_step failed with code {rc}")

        return {
            "v_cmd": float(step_out.v_cmd),
            "w_cmd": float(step_out.w_cmd),
            "state": int(step_out.state),
            "state_name": step_out.state_name.decode("utf-8"),
            "publish_stop": bool(step_out.publish_stop),
            "corridor_left_m": float(step_out.corridor_left_m),
            "corridor_right_m": float(step_out.corridor_right_m),
            "corridor_target_y_m": float(step_out.corridor_target_y_m),
            "lane_left_m": float(step_out.lane_left_m),
            "lane_right_m": float(step_out.lane_right_m),
            "speed_cap_mps": float(step_out.speed_cap_mps),
            "active_digit": int(step_out.active_digit),
            "digit_target_active": bool(step_out.digit_target_active),
            "obstacle_count": int(step_out.obstacle_count),
            "near_obstacle_count": int(step_out.near_obstacle_count),
            "far_obstacle_count": int(step_out.far_obstacle_count),
            "has_blocking_obstacle": bool(step_out.has_blocking_obstacle),
            "preferred_branch": int(step_out.preferred_branch),
            "active_mode_tag": step_out.active_mode_tag.decode("utf-8"),
            "cbf_active": bool(step_out.cbf_active),
            "cbf_min_clearance_m": float(step_out.cbf_min_clearance_m),
            "cbf_limiting_source": step_out.cbf_limiting_source.decode("utf-8"),
            "finish_armed": bool(step_out.finish_armed),
            "permanent_stop": bool(step_out.permanent_stop),
            "yellow_stop_active": bool(step_out.yellow_stop_active),
            "yellow_brake_active": bool(step_out.yellow_brake_active),
            "yellow_count": int(step_out.yellow_count),
            "yellow_dist_valid": bool(step_out.yellow_dist_valid),
            "yellow_is_finish": bool(step_out.yellow_is_finish),
            "yellow_rearmed": bool(step_out.yellow_rearmed),
            "yellow_d_bumper_m": float(step_out.yellow_d_bumper_m),
            "yellow_phase": step_out.yellow_phase.decode("utf-8"),
            "total_arc_m": float(step_out.total_arc_m),
            "friction_scale": float(step_out.friction_scale),
            "effective_brake_decel": float(step_out.effective_brake_decel),
            "corridor_left_px": int(step_out.corridor_left_px),
            "corridor_right_px": int(step_out.corridor_right_px),
            "corridor_ref_px": int(step_out.corridor_ref_px),
            "corridor_row_y": int(step_out.corridor_row_y),
        }
