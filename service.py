from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

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

app = Flask(__name__)

DEFAULT_ROBOT_ID = "0"
coordinator: LocalizationCoordinator | None = None
app_config: Dict[str, Any] | None = None
default_map_id: str | None = None
_robot_request_locks: dict[str, threading.Lock] = {}
_robot_request_locks_guard = threading.Lock()


def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    global coordinator, app_config, default_map_id
    with open(config_path, "r", encoding="utf-8") as file:
        app_config = yaml.safe_load(file)

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
        ),
    )
    depth_nav_cfg = app_config.get("depth_nav", {})
    pose_pipeline = PosePipeline(
        model=pose_model,
        config=PosePipelineConfig(
            depth_nav_enabled=depth_nav_cfg.get("able", False),
            height_up=depth_nav_cfg.get("height_up", "z"),
            camera_real_h=depth_nav_cfg.get("camera_real_h"),
            min_dist=depth_nav_cfg.get("min_dist", 0.1),
            max_dist=depth_nav_cfg.get("max_dist", 5.0),
        ),
    )
    coordinator = LocalizationCoordinator(
        model_manager=model_manager,
        map_manager=map_manager,
        session_manager=session_manager,
        vpr_pipeline=vpr_pipeline,
        pose_pipeline=pose_pipeline,
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
        )

    check_ref = app_config["vpr"].get("check_ref", False)
    for item in map_defs:
        coordinator.prepare_map(item["id"], check_ref=check_ref)

    default_map_id = map_defs[0]["id"]
    session_manager.bind_map(DEFAULT_ROBOT_ID, default_map_id)
    logging.info("VPS system initialized successfully")
    return app


def _resolve_map_definitions(config: Dict[str, Any]) -> List[Dict[str, Path | str | None]]:
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

        if "ref_data_path" not in item or "ref_descriptors_path" not in item:
            raise RuntimeError(
                f"Map '{map_id}' must define both ref_data_path and ref_descriptors_path."
            )

        root_dir = Path(item["ref_data_path"])
        definitions.append(
            {
                "id": map_id,
                "root_dir": root_dir,
                "ref_descriptors_path": Path(item["ref_descriptors_path"]),
                "depth_dir": Path(item["depth_dir"]) if item.get("depth_dir") else (root_dir / "depth"),
                "calibration_dir": Path(item["calibration_dir"]) if item.get("calibration_dir") else (root_dir / "calibration"),
                "render_rgb_dir": Path(item["render_rgb_dir"]) if item.get("render_rgb_dir") else (root_dir / "rgb_render"),
                "render_depth_dir": Path(item["render_depth_dir"]) if item.get("render_depth_dir") else (root_dir / "depth_render"),
                "nav_map_path": Path(item["nav_map_path"]) if item.get("nav_map_path") else None,
                "nav_yaml_path": Path(item["nav_yaml_path"]) if item.get("nav_yaml_path") else None,
            }
        )
    return definitions


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
        robot_id = request.form.get("robot_id", DEFAULT_ROBOT_ID)
        requested_map_id = request.form.get("map_id")
        request_tag = f"robot={robot_id}"

        logging.info(f"[{request_tag}] rgb_file={rgb_file.filename}")
        if rgb_file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        allowed_extensions = app_config["service"]["supported_formats"]
        if not any(rgb_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": f"Unsupported file format. Supported: {allowed_extensions}"}), 400

        request_lock = _get_robot_request_lock(robot_id)
        logging.info(f"[{request_tag}] request_parse_time={time.time() - start_time:.6f}s")

        lock_wait_start = time.time()
        with request_lock:
            logging.info(f"[{request_tag}] lock_wait_time={time.time() - lock_wait_start:.6f}s")
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
            localize_start = time.time()
            result = coordinator.localize(
                robot_id=robot_id,
                query_image=Path(query_image_path),
                depth_paths=[Path(depth_path)] if depth_path else None,
            )
            logging.info(f"[{request_tag}] coordinator_localize_time={time.time() - localize_start:.6f}s")
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
        logging.info(f"[{request_tag}] map_response_time={time.time() - start_time:.6f}s")
        logging.info(f"[{request_tag}] service_total_time={time.time() - start_time:.6f}s")
        return jsonify(response_data), 200
    except Exception as exc:
        logging.error(f"Error during localization: {str(exc)}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    os.makedirs("./log", exist_ok=True)
    log_filename = datetime.datetime.now().strftime("log/log_%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s"
    )
    app = create_app()
    app.run(host="0.0.0.0", port=6001, debug=False)
