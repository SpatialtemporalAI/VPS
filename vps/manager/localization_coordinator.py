from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from vps.manager.map_manager import MapManager
from vps.manager.model_manager import ModelManager
from vps.manager.session_manager import SessionManager
from vps.maps.map_instance import MapInstance
from vps.maps.pose_map import PoseMap, PoseMapConfig
from vps.maps.vpr_map import VPRMap, VPRMapConfig
from vps.pipeline.pose_pipeline import LocalizationResult, PosePipeline, PosePipelineInput
from vps.pipeline.vpr_pipeline import VPRPipeline
import time
import logging
@dataclass
class CoordinatorModels:
    vpr_model_name: str = "vpr"
    pose_model_name: str = "pose"


class LocalizationCoordinator:
    """Top-level orchestrator for map/session/model/pipeline flow."""

    def __init__(
        self,
        model_manager: ModelManager,
        map_manager: MapManager,
        session_manager: SessionManager,
        vpr_pipeline: VPRPipeline,
        pose_pipeline: PosePipeline,
        models: Optional[CoordinatorModels] = None,
    ):
        self.model_manager = model_manager
        self.map_manager = map_manager
        self.session_manager = session_manager
        self.vpr_pipeline = vpr_pipeline
        self.pose_pipeline = pose_pipeline
        self.models = models or CoordinatorModels()
        self.result_map_dir = Path("data/outputs/result_maps")
        self._result_map_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="result-map",
        )

    def register_map(
        self,
        map_id: str,
        root_dir: Path,
        ref_descriptors_path: Path,
        *,
        depth_dir: Optional[Path] = None,
        calibration_dir: Optional[Path] = None,
        render_rgb_dir: Optional[Path] = None,
        render_depth_dir: Optional[Path] = None,
        nav_map_path: Optional[Path] = None,
        nav_yaml_path: Optional[Path] = None,
    ) -> None:
        root_dir = Path(root_dir)
        vpr_map = VPRMap(
            VPRMapConfig(
                map_id=map_id,
                root_dir=root_dir,
                rgb_dir=root_dir / "rgb",
                poses_dir=root_dir / "poses",
                ref_descriptors_path=Path(ref_descriptors_path),
            )
        )
        pose_map = PoseMap(
            PoseMapConfig(
                map_id=map_id,
                root_dir=root_dir,
                poses_dir=root_dir / "poses",
                depth_dir=depth_dir,
                render_rgb_dir=render_rgb_dir,
                render_depth_dir=render_depth_dir,
                calibration_dir=calibration_dir,
                nav_map_path=nav_map_path,
                nav_yaml_path=nav_yaml_path,
            )
        )
        self.map_manager.register(
            MapInstance(
                map_id=map_id,
                vpr_map=vpr_map,
                pose_map=pose_map,
            )
        )

    def prepare_map(self, map_id: str, check_ref: bool = False) -> None:
        vpr_model = self.model_manager.get(self.models.vpr_model_name)
        self.map_manager.prepare_map(map_id, vpr_model=vpr_model, check_ref=check_ref)
        self._prepare_pose_map(map_id)

    def switch_map(self, robot_id: str, map_id: str, check_ref: bool = False) -> None:
        self.prepare_map(map_id, check_ref=check_ref)
        self.session_manager.bind_map(robot_id, map_id)

    def localize(
        self,
        robot_id: str,
        query_image: Path,
        depth_paths: Optional[list[Path]] = None,
        poses_paths: Optional[list[Path]] = None,
        k_paths: Optional[list[Path]] = None,
        last_pose_hint: Optional[np.ndarray] = None,
    ) -> LocalizationResult:
        start_time = time.time()
        request_tag = f"robot={robot_id}"
        session = self.session_manager.get_or_create(robot_id)
        if session.active_map_id is None:
            raise RuntimeError(f"Robot '{robot_id}' has no active map. Call switch_map first.")

        map_instance = self.map_manager.get(session.active_map_id)
        last_pose_xyz = self._pose_to_xyz(last_pose_hint)
        if last_pose_xyz is None:
            last_pose_xyz = self._pose_to_xyz(session.last_pose)
        vpr_start = time.time()
        vpr_result = self.vpr_pipeline.run(
            query_image=Path(query_image),
            vpr_map=map_instance.vpr_map,
            last_pose=last_pose_xyz,
        )
        logging.info(
            f"[{request_tag}] vpr_pipeline_time={time.time() - vpr_start:.6f}s "
            f"map_id={session.active_map_id} num_refs={len(vpr_result.ref_image_paths)}"
        )

        pose_start = time.time()
        pose_result = self.pose_pipeline.run(
            PosePipelineInput(
                query_image=vpr_result.query_image_path,
                ref_image_paths=vpr_result.ref_image_paths,
                depth_paths=depth_paths,
                poses_paths=poses_paths,
                K_paths=k_paths,
            ),
            pose_map=map_instance.pose_map,
        )
        if pose_result.pose_c2w is not None:
            session.update_pose(pose_result.pose_c2w)
            self._schedule_result_map_save(
                robot_id=robot_id,
                pose_map=map_instance.pose_map,
                pose_result=pose_result,
            )
        logging.info(
            f"[{request_tag}] pose_pipeline_time={time.time() - pose_start:.6f}s "
            f"success={pose_result.pose_c2w is not None}"
        )
        logging.info(f"[{request_tag}] coordinator_total_time={time.time() - start_time:.6f}s")
        return pose_result

    def _prepare_pose_map(self, map_id: str) -> None:
        map_instance = self.map_manager.get(map_id)
        pose_map = map_instance.pose_map
        if pose_map.is_loaded():
            return

        poses_dir = pose_map.config.poses_dir
        rgb_dir = map_instance.vpr_map.config.rgb_dir
        if not poses_dir.exists():
            raise FileNotFoundError(f"Pose directory does not exist: {poses_dir}")
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")

        pose_by_stem: dict[str, np.ndarray] = {}
        for pose_path in poses_dir.glob("*.txt"):
            pose_by_stem[pose_path.stem] = np.loadtxt(pose_path).reshape(4, 4)

        ref_pose_map: dict[str, np.ndarray] = {}
        for image_path in rgb_dir.iterdir():
            if not image_path.is_file():
                continue
            stem = image_path.stem
            pose = pose_by_stem.get(stem)
            if pose is not None:
                ref_pose_map[image_path.name] = pose

        if not ref_pose_map:
            raise RuntimeError(f"No valid pose data found for map '{map_id}' in {poses_dir}")
        pose_map.load(ref_pose_map)

    @staticmethod
    def _pose_to_xyz(pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pose is None:
            return None
        pose = np.asarray(pose)
        if pose.shape == (4, 4):
            return pose[:3, 3]
        if pose.ndim == 1 and pose.shape[0] >= 3:
            return pose[:3]
        return None

    def _schedule_result_map_save(
        self,
        robot_id: str,
        pose_map: PoseMap,
        pose_result: LocalizationResult,
    ) -> None:
        self._result_map_executor.submit(
            self._save_result_map,
            robot_id,
            pose_map,
            pose_result,
        )

    def _save_result_map(
        self,
        robot_id: str,
        pose_map: PoseMap,
        pose_result: LocalizationResult,
    ) -> None:
        try:
            nav_context = pose_map.build_nav_context()
            map_path = nav_context.map_path
            yaml_path = nav_context.yaml_path
            if map_path is None or yaml_path is None:
                return
            if not map_path.exists() or not yaml_path.exists():
                return

            if pose_result.pose_c2w is None:
                return

            base_map = pose_result.occupancy_map
            if base_map is None:
                base_map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
            if base_map is None:
                return

            result_map = self._render_pose_on_map(
                pose_c2w=pose_result.pose_c2w,
                yaml_path=yaml_path,
                base_map=base_map,
            )
            if result_map is None:
                return

            self.result_map_dir.mkdir(parents=True, exist_ok=True)
            result_path = self.result_map_dir / f"id{robot_id}_result.png"
            if not cv2.imwrite(str(result_path), result_map):
                logging.warning("Failed to save result map to %s", result_path)
        except Exception:
            logging.exception("Asynchronous result-map save failed.")

    @staticmethod
    def _render_pose_on_map(
        pose_c2w: np.ndarray,
        yaml_path: Path,
        base_map: np.ndarray,
    ) -> Optional[np.ndarray]:
        if base_map is None:
            return None

        with open(yaml_path, "r", encoding="utf-8") as file:
            map_metadata = yaml.safe_load(file)

        resolution = map_metadata.get("resolution")
        origin_world = map_metadata.get("origin")
        if resolution is None or origin_world is None:
            return None

        if base_map.ndim == 2:
            colored_map = cv2.cvtColor(base_map.copy(), cv2.COLOR_GRAY2BGR)
        elif base_map.ndim == 3:
            colored_map = base_map.copy()
        else:
            return None

        map_rows, map_cols = colored_map.shape[:2]
        x_min_world = origin_world[0]
        y_min_world = origin_world[1]

        grid_x_cam = int((pose_c2w[0, 3] - x_min_world) / resolution)
        row_idx_from_origin_bottom_cam = int((pose_c2w[1, 3] - y_min_world) / resolution)
        grid_y_cam = map_rows - 1 - row_idx_from_origin_bottom_cam

        if not (0 <= grid_x_cam < map_cols and 0 <= grid_y_cam < map_rows):
            return colored_map

        radius = max(2, int(min(map_rows, map_cols) / 200))
        cv2.circle(
            colored_map,
            (grid_x_cam, grid_y_cam),
            radius,
            (0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        return colored_map
