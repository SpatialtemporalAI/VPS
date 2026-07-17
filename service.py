from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request

from vps.manager import (
    LocalizationCoordinator,
    MapManager,
    ModelManager,
    SessionManager,
)
from vps.models import PoseModel, VPRModel
from vps.pipeline import (
    PosePipeline,
    PosePipelineConfig,
    VPRPipeline,
    VPRPipelineConfig,
)
from vps.refinement import GsplatRefinementConfig, GsplatRefinementPipeline
from vps.utils.trajectory_filter import TrajectoryFilter, TrajectoryFilterConfig


ROBOT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


class FairInferenceGate:
    """A process-local FIFO gate for shared GPU inference resources."""

    def __init__(self, capacity: int = 1):
        if capacity < 1:
            raise ValueError("FairInferenceGate capacity must be at least 1.")
        self._capacity = capacity
        self._in_use = 0
        self._queue: deque[object] = deque()
        self._condition = threading.Condition()

    def acquire(self) -> None:
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            while self._queue[0] is not ticket or self._in_use >= self._capacity:
                self._condition.wait()
            self._queue.popleft()
            self._in_use += 1

    def release(self) -> None:
        with self._condition:
            if self._in_use <= 0:
                raise RuntimeError("FairInferenceGate released without a matching acquire.")
            self._in_use -= 1
            self._condition.notify_all()


app = Flask(__name__)

DEFAULT_ROBOT_ID = "0"
coordinator: LocalizationCoordinator | None = None
app_config: Dict[str, Any] | None = None
default_map_id: str | None = None
default_logical_map_id: str | None = None
light_map_groups: Dict[str, Dict[str, str]] = {}
_robot_request_locks: dict[str, threading.Lock] = {}
_robot_request_locks_guard = threading.Lock()
_inference_gate = FairInferenceGate(capacity=1)


def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    global coordinator, app_config, default_map_id, default_logical_map_id, light_map_groups
    with open(config_path, "r", encoding="utf-8") as file:
        app_config = yaml.safe_load(file)
    light_map_groups = {}

    model_manager = ModelManager()
    map_manager = MapManager()
    session_manager = SessionManager()

    vpr_model = VPRModel(app_config)
    pose_model = PoseModel(app_config)
    model_manager.register("vpr", vpr_model)
    model_manager.register("pose", pose_model)

    vpr_pipeline = VPRPipeline(
        model=vpr_model,
        config=VPRPipelineConfig(
            top_k=app_config["vpr"]["top_k"],
            size_num_matched=app_config["vpr"].get("size_num_matched", 1),
            similarity_threshold=app_config["vpr"].get("similarity_threshold"),
            use_spatial_filtering=app_config["vpr"].get("use_spatial_filtering", False),
            spatial_radius=app_config["vpr"].get("spatial_radius"),
            use_single_query_fast_path=app_config["vpr"].get("use_single_query_fast_path", True),
        ),
    )
    depth_nav_cfg = app_config.get("depth_nav", {})
    omega_temporal_cfg = (
        app_config.get("pose", {})
        .get("vggt_omega", {})
        .get("temporal_window", {})
    )
    pose_pipeline = PosePipeline(
        model=pose_model,
        config=PosePipelineConfig(
            depth_nav_enabled=depth_nav_cfg.get("able", False),
            height_up=depth_nav_cfg.get("height_up", "z"),
            camera_real_h=depth_nav_cfg.get("camera_real_h"),
            min_dist=depth_nav_cfg.get("min_dist", 0.1),
            max_dist=depth_nav_cfg.get("max_dist", 5.0),
            temporal_motion_max_frames=max(
                0,
                int(omega_temporal_cfg.get("motion_max_frames", 3)),
            ),
        ),
    )
    trajectory_filter_cfg = app_config.get("trajectory_filter", {})
    max_yaw_rate_degps = trajectory_filter_cfg.get("max_yaw_rate_degps")
    if max_yaw_rate_degps is not None:
        max_yaw_rate_degps = float(max_yaw_rate_degps)
    refinement_pipeline = None
    gsplat_refine_cfg = app_config.get("gsplat_refinement", {})
    if gsplat_refine_cfg.get("enabled", False):
        refinement_pipeline = GsplatRefinementPipeline(
            GsplatRefinementConfig(
                enabled=True,
                matcher=str(gsplat_refine_cfg.get("matcher", "superpoint_lightglue")),
                min_matches=int(gsplat_refine_cfg.get("min_matches", 80)),
                min_pnp_inliers=int(gsplat_refine_cfg.get("min_pnp_inliers", 30)),
                max_translation_delta_m=float(
                    gsplat_refine_cfg.get("max_translation_delta_m", 1.0)
                ),
                max_yaw_delta_deg=float(gsplat_refine_cfg.get("max_yaw_delta_deg", 45.0)),
                alpha_threshold=float(gsplat_refine_cfg.get("alpha_threshold", 0.1)),
                min_depth=float(gsplat_refine_cfg.get("min_depth", 0.05)),
                max_depth=float(gsplat_refine_cfg.get("max_depth", 100.0)),
                ransac_reproj_error=float(
                    gsplat_refine_cfg.get("ransac_reproj_error", 4.0)
                ),
                ransac_confidence=float(gsplat_refine_cfg.get("ransac_confidence", 0.999)),
                ransac_iterations=int(gsplat_refine_cfg.get("ransac_iterations", 1000)),
                max_keypoints=int(gsplat_refine_cfg.get("max_keypoints", 4096)),
                detection_threshold=float(
                    gsplat_refine_cfg.get("detection_threshold", 0.0005)
                ),
                save_render=bool(gsplat_refine_cfg.get("save_render", False)),
                render_camera=gsplat_refine_cfg.get("render_camera"),
                query_camera=gsplat_refine_cfg.get("query_camera"),
                device=str(gsplat_refine_cfg.get("device", app_config["system"]["device"])),
            )
        )
    coordinator = LocalizationCoordinator(
        model_manager=model_manager,
        map_manager=map_manager,
        session_manager=session_manager,
        vpr_pipeline=vpr_pipeline,
        pose_pipeline=pose_pipeline,
        refinement_pipeline=refinement_pipeline,
        trajectory_filter=TrajectoryFilter(
            TrajectoryFilterConfig(
                enabled=trajectory_filter_cfg.get("enabled", True),
                max_speed_mps=float(trajectory_filter_cfg.get("max_speed_mps", 2.0)),
                speed_margin_m=float(trajectory_filter_cfg.get("speed_margin_m", 0.5)),
                min_time_delta_seconds=float(
                    trajectory_filter_cfg.get("min_time_delta_seconds", 0.1)
                ),
                max_yaw_rate_degps=max_yaw_rate_degps,
                yaw_margin_deg=float(trajectory_filter_cfg.get("yaw_margin_deg", 30.0)),
                reject_invalid_pose=trajectory_filter_cfg.get("reject_invalid_pose", True),
                temporal_min_translation_m=float(
                    trajectory_filter_cfg.get("temporal_min_translation_m", 0.15)
                ),
                temporal_min_yaw_deg=float(
                    trajectory_filter_cfg.get("temporal_min_yaw_deg", 8.0)
                ),
                temporal_min_interval_seconds=float(
                    trajectory_filter_cfg.get("temporal_min_interval_seconds", 0.5)
                ),
            )
        ),
    )

    map_defs = _resolve_map_definitions(app_config)
    if not map_defs:
        raise RuntimeError("No map definitions found in config.")

    for item in map_defs:
        coordinator.register_map(
            map_id=item["id"],
            root_dir=item["root_dir"],
            ref_descriptors_path=item["ref_descriptors_path"],
            depth_dir=item.get("depth_dir"),
            calibration_dir=item.get("calibration_dir"),
            render_rgb_dir=item.get("render_rgb_dir"),
            render_depth_dir=item.get("render_depth_dir"),
            nav_map_path=item.get("nav_map_path"),
            nav_yaml_path=item.get("nav_yaml_path"),
            vggt_omega_ref_cache_path=item.get("vggt_omega_ref_cache_path"),
            gaussian_ply_path=item.get("gaussian_ply_path"),
            gaussian_camera=item.get("gaussian_camera"),
            query_camera=item.get("query_camera"),
        )

    check_ref = app_config["vpr"].get("check_ref", False)
    for item in map_defs:
        coordinator.prepare_map(item["id"], check_ref=check_ref)

    default_map_id = map_defs[0]["id"]
    default_logical_map_id = str(app_config["service"]["maps"][0].get("id", default_map_id))
    session_manager.bind_map(DEFAULT_ROBOT_ID, default_map_id)
    logging.info("VPS system initialized successfully")
    return app


def _resolve_map_definitions(config: Dict[str, Any]) -> List[Dict[str, Path | str | None]]:
    global light_map_groups
    service_maps = config.get("service", {}).get("maps")
    if not service_maps:
        raise RuntimeError(
            "Missing required `service.maps` configuration. "
            "Map paths are now strictly configured under `service.maps`."
        )

    definitions: List[Dict[str, Path | str | None]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(service_maps):
        map_id = str(item.get("id", f"map_{idx}"))
        if map_id in seen_ids:
            raise RuntimeError(f"Duplicated map id in service.maps: {map_id}")
        seen_ids.add(map_id)

        light_variants = item.get("light_variants")
        if light_variants:
            if not isinstance(light_variants, dict):
                raise RuntimeError(f"Map '{map_id}' light_variants must be a mapping.")
            variant_ids: Dict[str, str] = {}
            for raw_variant_name, variant_item in light_variants.items():
                variant_name = str(raw_variant_name).lower()
                expanded_map_id = f"{map_id}__{variant_name}"
                if expanded_map_id in seen_ids:
                    raise RuntimeError(f"Duplicated expanded map id in service.maps: {expanded_map_id}")
                seen_ids.add(expanded_map_id)
                definitions.append(
                    _build_map_definition(
                        map_id=expanded_map_id,
                        item=variant_item,
                        parent_map_id=map_id,
                    )
                )
                variant_ids[variant_name] = expanded_map_id
            light_map_groups[map_id] = variant_ids
            continue

        definitions.append(_build_map_definition(map_id=map_id, item=item))
    return definitions


def _build_map_definition(
    map_id: str,
    item: Dict[str, Any],
    parent_map_id: str | None = None,
) -> Dict[str, Path | str | None]:
    if "ref_data_path" not in item or "ref_descriptors_path" not in item:
        if parent_map_id is None:
            raise RuntimeError(
                f"Map '{map_id}' must define both ref_data_path and ref_descriptors_path."
            )
        raise RuntimeError(
            f"Map '{parent_map_id}' variant '{map_id}' must define both ref_data_path and ref_descriptors_path."
        )

    root_dir = Path(item["ref_data_path"])
    return {
        "id": map_id,
        "root_dir": root_dir,
        "ref_descriptors_path": Path(item["ref_descriptors_path"]),
        "depth_dir": Path(item["depth_dir"]) if item.get("depth_dir") else (root_dir / "depth"),
        "calibration_dir": Path(item["calibration_dir"]) if item.get("calibration_dir") else (root_dir / "calibration"),
        "render_rgb_dir": Path(item["render_rgb_dir"]) if item.get("render_rgb_dir") else (root_dir / "rgb_render"),
        "render_depth_dir": Path(item["render_depth_dir"]) if item.get("render_depth_dir") else (root_dir / "depth_render"),
        "nav_map_path": Path(item["nav_map_path"]) if item.get("nav_map_path") else None,
        "nav_yaml_path": Path(item["nav_yaml_path"]) if item.get("nav_yaml_path") else None,
        "vggt_omega_ref_cache_path": Path(item["vggt_omega_ref_cache_path"]) if item.get("vggt_omega_ref_cache_path") else None,
        "gaussian_ply_path": Path(item["gaussian_ply_path"]) if item.get("gaussian_ply_path") else None,
        "gaussian_camera": item.get("gaussian_camera"),
        "query_camera": item.get("query_camera"),
    }


def get_robot_to_camera_transform(offset_z: float = 0):
    T = np.eye(4)
    T[2, 3] = offset_z
    return T


def transform_matrix_to_pose_2d(transform_matrix):
    if transform_matrix is None:
        return None
    transform_matrix = transform_matrix @ get_robot_to_camera_transform()
    translation = transform_matrix[:3, 3]
    x, y, _ = translation[0], translation[1], translation[2]
    rotation_matrix = transform_matrix[:3, :3]
    theta = np.arctan2(rotation_matrix[0, 0], -1 * rotation_matrix[1, 0])
    return {
        "x": float(x),
        "y": float(y),
        "theta": float(theta),
    }


def _get_robot_request_lock(robot_id: str) -> threading.Lock:
    with _robot_request_locks_guard:
        lock = _robot_request_locks.get(robot_id)
        if lock is None:
            lock = threading.Lock()
            _robot_request_locks[robot_id] = lock
        return lock


def _validate_robot_id(robot_id: str) -> str | None:
    if ROBOT_ID_PATTERN.fullmatch(robot_id):
        return robot_id
    return None


def _run_with_robot_request_lock(
    robot_id: str,
    request_tag: str,
    operation: Callable[[], Any],
):
    """Reject a newer frame from the same robot while its prior frame is active."""
    request_lock = _get_robot_request_lock(robot_id)
    if not request_lock.acquire(blocking=False):
        logging.info("[%s] rejected status=busy", request_tag)
        response = jsonify(
            {
                "status": "busy",
                "robot_id": robot_id,
                "message": "previous request for this robot is still running",
            }
        )
        response.headers["Retry-After"] = "1"
        return response, 429
    try:
        return operation()
    finally:
        request_lock.release()


def _ensure_robot_map_binding(robot_id: str, requested_map_id: str | None) -> None:
    assert coordinator is not None
    assert default_map_id is not None
    session = coordinator.session_manager.get_or_create(robot_id)
    target_map = requested_map_id or session.active_map_id or default_map_id
    if target_map is None:
        raise RuntimeError("No target map is available.")
    if requested_map_id and requested_map_id != session.active_map_id:
        if not coordinator.map_manager.is_loaded(requested_map_id):
            coordinator.prepare_map(requested_map_id, check_ref=False)
        coordinator.session_manager.bind_map(robot_id, requested_map_id)
        return
    if session.active_map_id is None:
        coordinator.session_manager.bind_map(robot_id, target_map)


def _measure_image_brightness(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read query image for light detection: {image_path}")
    return float(np.mean(image))


def _select_light_state_with_history(
    robot_id: str,
    logical_map_id: str,
    mean_brightness: float,
    light_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    assert coordinator is not None
    session = coordinator.session_manager.get_or_create(robot_id)
    light_runtime = session.runtime_overrides.setdefault("light_selector", {})
    map_runtime = light_runtime.setdefault(logical_map_id, {})

    base_threshold = float(light_cfg.get("brightness_threshold", 110.0))
    dark_threshold_low = float(light_cfg.get("dark_threshold_low", base_threshold - 10.0))
    bright_threshold_high = float(light_cfg.get("bright_threshold_high", base_threshold + 10.0))
    switch_confirm_frames = max(1, int(light_cfg.get("switch_confirm_frames", 3)))

    previous_state = map_runtime.get("current_state")
    previous_candidate_state = map_runtime.get("candidate_state")
    previous_candidate_count = int(map_runtime.get("candidate_count", 0))
    previous_brightness = map_runtime.get("last_brightness")

    current_state = previous_state
    if current_state is None:
        midpoint = (dark_threshold_low + bright_threshold_high) / 2.0
        current_state = "bright" if mean_brightness >= midpoint else "dark"
        map_runtime["current_state"] = current_state
        map_runtime["candidate_state"] = None
        map_runtime["candidate_count"] = 0
        map_runtime["last_brightness"] = mean_brightness
        decision = {
            "selected_state": current_state,
            "observed_state": current_state,
            "candidate_state": None,
            "candidate_count": 0,
            "switched": False,
            "dark_threshold_low": dark_threshold_low,
            "bright_threshold_high": bright_threshold_high,
            "switch_confirm_frames": switch_confirm_frames,
            "initialized": True,
            "previous_state": previous_state,
            "previous_candidate_state": previous_candidate_state,
            "previous_candidate_count": previous_candidate_count,
            "previous_brightness": previous_brightness,
        }
        logging.info(
            "[robot=%s|light_history] logical_map=%s initialized state=%s mean_brightness=%.3f midpoint=%.3f",
            robot_id,
            logical_map_id,
            current_state,
            mean_brightness,
            midpoint,
        )
        return decision

    if mean_brightness <= dark_threshold_low:
        observed_state = "dark"
    elif mean_brightness >= bright_threshold_high:
        observed_state = "bright"
    else:
        observed_state = current_state

    candidate_state = previous_candidate_state
    candidate_count = previous_candidate_count
    switched = False

    if observed_state == current_state:
        candidate_state = None
        candidate_count = 0
    else:
        if candidate_state == observed_state:
            candidate_count += 1
        else:
            candidate_state = observed_state
            candidate_count = 1

        if candidate_count >= switch_confirm_frames:
            current_state = observed_state
            switched = True
            candidate_state = None
            candidate_count = 0

    map_runtime["current_state"] = current_state
    map_runtime["candidate_state"] = candidate_state
    map_runtime["candidate_count"] = candidate_count
    map_runtime["last_brightness"] = mean_brightness

    decision = {
        "selected_state": current_state,
        "observed_state": observed_state,
        "candidate_state": candidate_state,
        "candidate_count": candidate_count,
        "switched": switched,
        "dark_threshold_low": dark_threshold_low,
        "bright_threshold_high": bright_threshold_high,
        "switch_confirm_frames": switch_confirm_frames,
        "initialized": False,
        "previous_state": previous_state,
        "previous_candidate_state": previous_candidate_state,
        "previous_candidate_count": previous_candidate_count,
        "previous_brightness": previous_brightness,
    }
    logging.info(
        "[robot=%s|light_history] logical_map=%s mean_brightness=%.3f prev_state=%s observed_state=%s selected_state=%s prev_candidate=%s prev_candidate_count=%d candidate_state=%s candidate_count=%d switched=%s dark_threshold_low=%.3f bright_threshold_high=%.3f confirm_frames=%d prev_brightness=%s",
        robot_id,
        logical_map_id,
        mean_brightness,
        previous_state,
        observed_state,
        current_state,
        previous_candidate_state,
        previous_candidate_count,
        candidate_state,
        candidate_count,
        switched,
        dark_threshold_low,
        bright_threshold_high,
        switch_confirm_frames,
        f"{previous_brightness:.3f}" if isinstance(previous_brightness, (int, float)) else previous_brightness,
    )
    return decision


def _resolve_light_map_id(
    requested_map_id: str | None,
    detected_light_state: str,
) -> tuple[str, str | None]:
    assert default_map_id is not None
    assert default_logical_map_id is not None
    logical_map_id = requested_map_id or default_logical_map_id
    variant_group = light_map_groups.get(logical_map_id)
    if not variant_group:
        return logical_map_id, None

    preferred_names = [detected_light_state]
    if detected_light_state == "bright":
        preferred_names.append("light")
    elif detected_light_state == "dark":
        preferred_names.append("night")

    for variant_name in preferred_names:
        resolved_map_id = variant_group.get(variant_name)
        if resolved_map_id is not None:
            return resolved_map_id, variant_name

    return next(iter(variant_group.values())), next(iter(variant_group.keys()))


def _run_localization_request(
    *,
    rgb_file,
    robot_id: str,
    requested_map_id: str | None,
    request_tag: str,
    start_time: float,
    extra_response_fields: Dict[str, Any] | None = None,
):
    """Run a request after its caller has acquired the per-robot request lock."""
    global coordinator, app_config
    assert coordinator is not None
    assert app_config is not None

    logging.info(f"[{request_tag}] request_parse_time={time.time() - start_time:.6f}s")

    os.makedirs(app_config["service"]["temp_dir"], exist_ok=True)
    rgb_name = f"{Path(app_config['service']['temp_rgb_name']).stem}_{robot_id}{Path(app_config['service']['temp_rgb_name']).suffix}"
    query_image_path = os.path.join(
        app_config["service"]["temp_dir"],
        rgb_name,
    )
    rgb_save_start = time.time()
    rgb_file.save(query_image_path)
    logging.info(f"[{request_tag}] rgb_save_time={time.time() - rgb_save_start:.6f}s path={query_image_path}")

    depth_path = None
    if "depth" in request.files:
        depth_file = request.files["depth"]
        if depth_file and depth_file.filename:
            logging.info(f"[{request_tag}] depth_file={depth_file.filename}")
            depth_process_start = time.time()
            file_ext = os.path.splitext(depth_file.filename)[1].lower()
            depth_name = f"{Path(app_config['service']['temp_depth_name']).stem}_{robot_id}{Path(app_config['service']['temp_depth_name']).suffix}"
            depth_path = os.path.join(
                app_config["service"]["temp_dir"],
                depth_name,
            )
            if file_ext == ".png":
                filestr = depth_file.read()
                npimg = np.frombuffer(filestr, np.uint8)
                depth_data_png = cv2.imdecode(npimg, cv2.IMREAD_ANYDEPTH)
                if depth_data_png is not None:
                    depth_data = depth_data_png.astype(np.float32) / 1000.0
                    np.save(depth_path, depth_data)
                    logging.info(f"[{request_tag}] depth_png_to_npy_saved={depth_path}")
            elif file_ext == ".npy":
                depth_file.save(depth_path)
                logging.info(f"[{request_tag}] depth_npy_saved={depth_path}")
            logging.info(f"[{request_tag}] depth_process_time={time.time() - depth_process_start:.6f}s")

    map_bind_start = time.time()
    _ensure_robot_map_binding(robot_id, requested_map_id)
    logging.info(f"[{request_tag}] map_binding_time={time.time() - map_bind_start:.6f}s map_id={requested_map_id}")

    logging.info(f"[{request_tag}] preprocess_total_time={time.time() - start_time:.6f}s")
    gpu_wait_start = time.time()
    _inference_gate.acquire()
    logging.info(f"[{request_tag}] gpu_queue_wait_time={time.time() - gpu_wait_start:.6f}s")
    try:
        localize_start = time.time()
        result = coordinator.localize(
            robot_id=robot_id,
            query_image=Path(query_image_path),
            depth_paths=[Path(depth_path)] if depth_path else None,
        )
        logging.info(f"[{request_tag}] coordinator_localize_time={time.time() - localize_start:.6f}s")
    finally:
        _inference_gate.release()
    logging.info(f"[{request_tag}] post_lock_total_time={time.time() - start_time:.6f}s")

    response_data: Dict[str, Any] = {}
    if result.pose_c2w is not None:
        response_data["pose"] = transform_matrix_to_pose_2d(result.pose_c2w)
    else:
        response_data["pose"] = None
    logging.info(f"[{request_tag}] pose_response_time={time.time() - start_time:.6f}s")

    occupancy_map = result.occupancy_map
    if occupancy_map is not None and isinstance(occupancy_map, np.ndarray):
        assert occupancy_map.dtype == np.uint8, "occupancy map 必须是 uint8"
        ok, buf = cv2.imencode(".png", occupancy_map)
        if ok:
            response_data["map_png_b64"] = base64.b64encode(buf).decode("ascii")
            response_data["map_format"] = "png"
        else:
            response_data["map_png_b64"] = None
            response_data["map_format"] = None
    else:
        response_data["map_png_b64"] = None
        response_data["map_format"] = None

    if extra_response_fields:
        response_data.update(extra_response_fields)
    logging.info(f"[{request_tag}] map_response_time={time.time() - start_time:.6f}s")
    logging.info(f"[{request_tag}] service_total_time={time.time() - start_time:.6f}s")
    return jsonify(response_data), 200


@app.route("/localize", methods=["POST"])
def localize():
    """Localize one robot request.

    Expected multipart form-data fields:
    - image: required query RGB image file (.jpg/.jpeg/.png)
    - robot_id: optional robot identifier; defaults to DEFAULT_ROBOT_ID
    - map_id: optional target map id; if omitted, uses the robot's active map or the default map
    - depth: optional depth file (.png in millimeters or .npy in meters)

    Response JSON:
    - pose: {"x", "y", "theta"} or None
    - map_png_b64: optional occupancy/result map encoded as PNG base64
    - map_format: "png" when map_png_b64 is present
    """
    global coordinator, app_config
    if coordinator is None or app_config is None:
        return jsonify({"error": "Service is not initialized"}), 500

    start_time = time.time()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        rgb_file = request.files["image"]
        robot_id = _validate_robot_id(request.form.get("robot_id", DEFAULT_ROBOT_ID))
        if robot_id is None:
            return jsonify({"error": "Invalid robot_id. Use [A-Za-z0-9_-]{1,64}."}), 400
        requested_map_id = request.form.get("map_id")
        request_tag = f"robot={robot_id}"

        logging.info(f"[{request_tag}] rgb_file={rgb_file.filename}")
        if rgb_file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        allowed_extensions = app_config["service"]["supported_formats"]
        if not any(rgb_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": f"Unsupported file format. Supported: {allowed_extensions}"}), 400

        return _run_with_robot_request_lock(
            robot_id,
            request_tag,
            lambda: _run_localization_request(
                rgb_file=rgb_file,
                robot_id=robot_id,
                requested_map_id=requested_map_id,
                request_tag=request_tag,
                start_time=start_time,
            ),
        )
    except Exception as exc:
        logging.error(f"Error during localization: {str(exc)}")
        return jsonify({"error": str(exc)}), 500


@app.route("/localize_by_light", methods=["POST"])
def localize_by_light():
    """Localize with a bright/dark database selector that keeps per-robot history.

    Expected multipart form-data fields:
    - image: required query RGB image file (.jpg/.jpeg/.png)
    - robot_id: optional robot identifier; defaults to DEFAULT_ROBOT_ID
    - map_id: optional logical map id; if omitted, uses the default map group
    - depth: optional depth file (.png in millimeters or .npy in meters)

    Map config:
    - A logical map can define `light_variants`
    - Supported variant keys for now: bright / dark
    - If the logical map has no light_variants, it falls back to the map itself
    """
    global coordinator, app_config
    if coordinator is None or app_config is None:
        return jsonify({"error": "Service is not initialized"}), 500

    start_time = time.time()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        rgb_file = request.files["image"]
        robot_id = _validate_robot_id(request.form.get("robot_id", DEFAULT_ROBOT_ID))
        if robot_id is None:
            return jsonify({"error": "Invalid robot_id. Use [A-Za-z0-9_-]{1,64}."}), 400
        logical_map_id = request.form.get("map_id")
        request_tag = f"robot={robot_id}|light"

        logging.info(f"[{request_tag}] rgb_file={rgb_file.filename}")
        if rgb_file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        allowed_extensions = app_config["service"]["supported_formats"]
        if not any(rgb_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": f"Unsupported file format. Supported: {allowed_extensions}"}), 400

        def run_light_aware_localization():
            temp_dir = Path(app_config["service"]["temp_dir"])
            temp_dir.mkdir(parents=True, exist_ok=True)
            light_probe_suffix = Path(rgb_file.filename).suffix or ".jpg"
            light_probe_path = temp_dir / f"query_{robot_id}{light_probe_suffix}"
            rgb_file.save(light_probe_path)

            light_cfg = app_config.get("service", {}).get("light_selector", {})
            target_logical_map_id = logical_map_id or default_logical_map_id
            mean_brightness = _measure_image_brightness(light_probe_path)
            light_decision = _select_light_state_with_history(
                robot_id=robot_id,
                logical_map_id=target_logical_map_id,
                mean_brightness=mean_brightness,
                light_cfg=light_cfg,
            )
            resolved_map_id, resolved_variant = _resolve_light_map_id(
                requested_map_id=logical_map_id,
                detected_light_state=light_decision["selected_state"],
            )
            logging.info(
                "[%s] light_detection mean_brightness=%.3f observed_state=%s selected_state=%s "
                "candidate_state=%s candidate_count=%d switched=%s dark_threshold_low=%.3f "
                "bright_threshold_high=%.3f logical_map=%s resolved_map=%s resolved_variant=%s",
                request_tag,
                mean_brightness,
                light_decision["observed_state"],
                light_decision["selected_state"],
                light_decision["candidate_state"],
                light_decision["candidate_count"],
                light_decision["switched"],
                light_decision["dark_threshold_low"],
                light_decision["bright_threshold_high"],
                target_logical_map_id,
                resolved_map_id,
                resolved_variant,
            )

            rgb_file.stream.seek(0)
            return _run_localization_request(
                rgb_file=rgb_file,
                robot_id=robot_id,
                requested_map_id=resolved_map_id,
                request_tag=request_tag,
                start_time=start_time,
                extra_response_fields={
                    "light_state": light_decision["selected_state"],
                    "observed_light_state": light_decision["observed_state"],
                    "mean_brightness": mean_brightness,
                    "logical_map_id": target_logical_map_id,
                    "resolved_map_id": resolved_map_id,
                    "resolved_light_variant": resolved_variant,
                    "light_candidate_state": light_decision["candidate_state"],
                    "light_candidate_count": light_decision["candidate_count"],
                    "light_switched": light_decision["switched"],
                },
            )

        return _run_with_robot_request_lock(
            robot_id,
            request_tag,
            run_light_aware_localization,
        )
    except Exception as exc:
        logging.error(f"Error during light-aware localization: {str(exc)}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    os.makedirs("./log", exist_ok=True)
    log_filename = datetime.datetime.now().strftime("log/log_%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s"
    )
    app = create_app()
    app.run(host="0.0.0.0", port=6001, debug=False)
