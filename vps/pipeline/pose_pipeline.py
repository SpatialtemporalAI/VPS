from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol
import logging

import numpy as np

from vps.maps.pose_map import PoseMap
from vps.models.pose_model_contract import PoseModelOutput
from vps.nav.point2map import (
    get_floor_height,
    get_height_scale,
    get_new_occupancy_map,
    segment_points_h,
)
from vps.utils.motion_averaging_raw import MotionAveraging
from vps.utils.processing import trans_point_cloud


@dataclass
class PosePipelineInput:
    query_image: Path
    ref_image_paths: List[Path]
    robot_id: Optional[str] = None
    ref_poses: Optional[List[np.ndarray]] = None
    depth_paths: Optional[List[Path]] = None
    poses_paths: Optional[List[Path]] = None
    K_paths: Optional[List[Path]] = None


@dataclass
class LocalizationResult:
    pose_c2w: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    occupancy_map: Optional[np.ndarray]


@dataclass
class PosePipelineConfig:
    depth_nav_enabled: bool = False
    height_up: str = "z"
    camera_real_h: Optional[float] = None
    min_dist: float = 0.1
    max_dist: float = 5.0
    occupancy_min_points_per_cell: int = 15
    nav_show_self: bool = True
    temporal_motion_max_frames: int = 3


class PoseModelProtocol(Protocol):
    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        ref_poses: Optional[List[np.ndarray]] = None,
        depth_paths: Optional[List[Path]] = None,  #[query, ref1, ref2, ...] 
        poses_paths: Optional[List[Path]] = None, # [query, ref1, ref2, ...]
        k_paths: Optional[List[Path]] = None,
        ref_cache_path: Optional[Path] = None,
        robot_id: Optional[str] = None,
        map_id: Optional[str] = None,
    ) -> object:
        ...


class PosePipeline:
    """Run pose estimation with built-in motion-averaging solve step."""

    def __init__(
        self,
        model: PoseModelProtocol,
        config: Optional[PosePipelineConfig] = None,
    ):
        self.model = model
        self.motion_averaging = MotionAveraging()
        self.config = config or PosePipelineConfig()
        self._up_axis_index = {"x": 0, "y": 1, "z": 2}

    def run(self, payload: PosePipelineInput, pose_map: PoseMap) -> LocalizationResult:
        ref_images = payload.ref_image_paths
        model_output = self.model.infer(
            query_image=payload.query_image,
            ref_images=ref_images,
            ref_poses=payload.ref_poses,
            depth_paths=payload.depth_paths,
            poses_paths=payload.poses_paths,
            k_paths=payload.K_paths,
            ref_cache_path=pose_map.config.vggt_omega_ref_cache_path,
            robot_id=payload.robot_id,
            map_id=pose_map.config.map_id,
        )
        localization = self._solve_motion_averaging(
            model_output=model_output,
            pose_map=pose_map,
            ref_image_paths=payload.ref_image_paths,
            ref_poses=payload.ref_poses,
        )
        if localization.pose_c2w is None:
            return localization

        localization.depth = self._extract_query_depth(model_output)
        localization.occupancy_map = self._run_depth_navigation(
            model_output=model_output,
            pose_map=pose_map,
            final_pose=localization.pose_c2w,
        )
        return localization

    def _solve_motion_averaging(
        self,
        model_output: PoseModelOutput,
        pose_map: PoseMap,
        ref_image_paths: List[Path],
        ref_poses: Optional[List[np.ndarray]] = None,
    ) -> LocalizationResult:
        try:
            if not ref_image_paths:
                return LocalizationResult(
                    pose_c2w=None,
                    depth=None,
                    occupancy_map=None
                )

            num_refs = len(ref_image_paths)
            expected = num_refs + 1  # query + refs
            if model_output.extrinsic is None:
                raise ValueError("Pose model output is missing required extrinsic predictions.")
            if model_output.extrinsic.shape[0] < expected:
                raise ValueError(
                    f"Model output has {model_output.extrinsic.shape[0]} views, expected at least {expected}."
                )

            query_c2w = model_output.extrinsic[0]  # c2w format 4x4 numpy array
            ref_pred_c2w = [model_output.extrinsic[i] for i in range(1, expected)]  # c2w format
            if ref_poses is not None:
                if len(ref_poses) != num_refs:
                    raise ValueError(
                        f"VPR returned {len(ref_poses)} ref poses, expected {num_refs}."
                    )
                ref_gt_c2w = [np.asarray(pose) for pose in ref_poses]
            else:
                ref_gt_c2w = [
                    pose_map.get_ref_pose(Path(ref_path).name) for ref_path in ref_image_paths
                ]

            temporal_available = len(model_output.temporal_poses)
            temporal_use_count = min(
                temporal_available,
                self.config.temporal_motion_max_frames,
            )
            temporal_offset = temporal_available - temporal_use_count
            temporal_gt_c2w = [
                np.asarray(pose)
                for pose in model_output.temporal_poses[temporal_offset:]
            ]
            temporal_start = expected + temporal_offset
            temporal_end = min(
                model_output.extrinsic.shape[0],
                temporal_start + len(temporal_gt_c2w),
            )
            temporal_pred_c2w = [
                model_output.extrinsic[i] for i in range(temporal_start, temporal_end)
            ]
            if len(temporal_pred_c2w) != len(temporal_gt_c2w):
                temporal_gt_c2w = temporal_gt_c2w[: len(temporal_pred_c2w)]

            all_gt_c2w = [*ref_gt_c2w, *temporal_gt_c2w]
            all_pred_c2w = [*ref_pred_c2w, *temporal_pred_c2w]

            q2r_poses = []
            for pred_ref_c2w in all_pred_c2w:
                q2r = np.linalg.inv(pred_ref_c2w) @ query_c2w
                trans_norm = np.linalg.norm(q2r[:3, 3])
                if trans_norm > 1e-9:
                    q2r[:3, 3] = q2r[:3, 3] / trans_norm
                q2r_poses.append(q2r)

            logging.info(
                "Motion averaging inputs: map_refs=%d temporal_refs=%d temporal_available=%d total_refs=%d",
                len(ref_gt_c2w),
                len(temporal_gt_c2w),
                temporal_available,
                len(all_gt_c2w),
            )

            final_pose = self.motion_averaging.motion_averaging(all_gt_c2w, q2r_poses)
            return LocalizationResult(
                pose_c2w=final_pose,
                depth=None,
                occupancy_map=None
            )
        except Exception:
            logging.exception("Motion averaging solve failed; returning empty localization result.")
            return LocalizationResult(
                pose_c2w=None,
                depth=None,
                occupancy_map=None
            )

    def _extract_query_depth(self, model_output: PoseModelOutput) -> Optional[np.ndarray]:
        if model_output.depth_map is None:
            return None
        if model_output.depth_map.ndim < 3 or model_output.depth_map.shape[0] == 0:
            return None
        return model_output.depth_map[0].squeeze()

    def _run_depth_navigation(
        self,
        model_output: PoseModelOutput,
        pose_map: PoseMap,
        final_pose: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not self.config.depth_nav_enabled:
            return None
        if model_output.point_map is None:
            logging.warning("Depth navigation skipped: pose model output is missing point_map.")
            return None

        nav_context = pose_map.build_nav_context()
        map_path = nav_context.map_path
        yaml_path = nav_context.yaml_path
        if map_path is None or yaml_path is None:
            logging.warning("Depth navigation skipped: navigation map path or yaml path is missing.")
            return None
        if not map_path.exists() or not yaml_path.exists():
            logging.warning(
                "Depth navigation skipped: navigation map assets do not exist. "
                "map_exists=%s yaml_exists=%s",
                map_path.exists(),
                yaml_path.exists(),
            )
            return None
        if self.config.camera_real_h is None:
            logging.warning("Depth navigation skipped: camera_real_h is not configured.")
            return None

        try:
            all_points = np.concatenate(
                [points.reshape(-1, 3) for points in model_output.point_map],
                axis=0,
            )
            world_points = trans_point_cloud(all_points, extrinsic_cam=final_pose)
            up = self.config.height_up.lower()
            if up not in self._up_axis_index:
                raise ValueError(f"Unsupported height_up axis: {self.config.height_up}")
            cam_pred_h = final_pose[self._up_axis_index[up], 3]
            pred_floor_h = get_floor_height(
                pcd=world_points,
                cam_pred_h=cam_pred_h,
                up=up,
            )
            scale = get_height_scale(
                cam_pred_h=cam_pred_h,
                height_peak=pred_floor_h,
                cam_real_h=self.config.camera_real_h,
            )

            query_points = model_output.point_map[0]
            scaled_query_points = trans_point_cloud(
                query_points,
                extrinsic_cam=final_pose,
                scale=scale,
            )
            pred_floor_h = cam_pred_h - scale * (cam_pred_h - pred_floor_h)
            obstacle_points, available_points = segment_points_h(
                scaled_query_points,
                floor_height=pred_floor_h,
                cam_h=cam_pred_h,
                up=up,
            )
            return get_new_occupancy_map(
                obs_points=obstacle_points,
                ava_points=available_points,
                map_path=str(map_path),
                yaml_path=str(yaml_path),
                camera_6dpose=final_pose,
                min_dist=self.config.min_dist,
                max_dist=self.config.max_dist,
                occupancy_min_points_per_cell=self.config.occupancy_min_points_per_cell,
                up=up,
                showself=self.config.nav_show_self,
            )
        except Exception:
            logging.exception("Depth navigation failed; returning no occupancy map.")
            return None
