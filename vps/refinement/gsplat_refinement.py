from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from vps.maps.pose_map import PoseMap
from vps.refinement.gaussian_renderer import GaussianCamera, GsplatGaussianRenderer
from vps.refinement.image_matcher import SuperPointLightGlueMatcher
from vps.refinement.pnp_solver import PnPSolver


@dataclass
class GsplatRefinementConfig:
    enabled: bool = False
    matcher: str = "superpoint_lightglue"
    min_matches: int = 80
    min_pnp_inliers: int = 30
    max_translation_delta_m: float = 1.0
    max_yaw_delta_deg: float = 45.0
    alpha_threshold: float = 0.1
    min_depth: float = 0.05
    max_depth: float = 100.0
    ransac_reproj_error: float = 4.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 1000
    max_keypoints: int = 4096
    detection_threshold: float = 0.0005
    save_render: bool = False
    render_camera: dict[str, Any] | None = None
    query_camera: dict[str, Any] | None = None
    device: str = "cuda"


@dataclass
class GsplatRefinementResult:
    pose_c2w: Optional[np.ndarray]
    accepted: bool
    reason: str
    num_matches: int = 0
    valid_depth_matches: int = 0
    pnp_inliers: int = 0
    reprojection_error: float | None = None


class GsplatRefinementPipeline:
    """Render-at-pose and refine with SuperPoint+LightGlue 2D-3D PnP."""

    def __init__(self, config: GsplatRefinementConfig):
        self.config = config
        self.renderer = GsplatGaussianRenderer(device=config.device)
        self.matcher = SuperPointLightGlueMatcher(
            device=config.device,
            max_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold,
        )
        self.pnp = PnPSolver(
            ransac_reproj_error=config.ransac_reproj_error,
            confidence=config.ransac_confidence,
            iterations_count=config.ransac_iterations,
        )

    def run(
        self,
        query_image: Path,
        initial_pose_c2w: np.ndarray,
        pose_map: PoseMap,
        robot_id: str,
    ) -> GsplatRefinementResult:
        start = time.time()
        gaussian_path = pose_map.config.gaussian_ply_path
        if gaussian_path is None:
            return GsplatRefinementResult(None, False, "missing_gaussian_ply")

        render_camera = GaussianCamera.from_mapping(
            pose_map.config.gaussian_camera,
            self.config.render_camera,
        )
        if render_camera is None:
            return GsplatRefinementResult(None, False, "missing_render_camera")
        query_camera = GaussianCamera.from_mapping(
            pose_map.config.query_camera,
            self.config.query_camera or pose_map.config.gaussian_camera or self.config.render_camera,
        )
        if query_camera is None:
            return GsplatRefinementResult(None, False, "missing_query_camera")

        try:
            render = self.renderer.render(gaussian_path, initial_pose_c2w, render_camera)
            if self.config.save_render:
                self._save_render_image(robot_id, "render", render.rgb)
            matches = self.matcher.match(query_image, render.rgb)
            if matches.query_points.shape[0] < self.config.min_matches:
                return self._log_result(
                    GsplatRefinementResult(
                        None,
                        False,
                        "not_enough_matches",
                        num_matches=int(matches.query_points.shape[0]),
                    ),
                    start,
                )

            object_points, image_points = self._build_2d3d_correspondences(
                render_points=matches.render_points,
                query_points=matches.query_points,
                depth=render.depth,
                alpha=render.alpha,
                pose_c2w=initial_pose_c2w,
                camera=render_camera,
            )
            if object_points.shape[0] < self.config.min_pnp_inliers:
                return self._log_result(
                    GsplatRefinementResult(
                        None,
                        False,
                        "not_enough_valid_depth",
                        num_matches=int(matches.query_points.shape[0]),
                        valid_depth_matches=int(object_points.shape[0]),
                    ),
                    start,
                )

            pnp_result = self.pnp.solve(
                object_points_world=object_points,
                image_points=image_points,
                camera_matrix=query_camera.intrinsic(),
            )
            if pnp_result.pose_c2w is None:
                return self._log_result(
                    GsplatRefinementResult(
                        None,
                        False,
                        "pnp_failed",
                        num_matches=int(matches.query_points.shape[0]),
                        valid_depth_matches=int(object_points.shape[0]),
                        pnp_inliers=pnp_result.inlier_count,
                    ),
                    start,
                )
            if pnp_result.inlier_count < self.config.min_pnp_inliers:
                return self._log_result(
                    GsplatRefinementResult(
                        None,
                        False,
                        "not_enough_pnp_inliers",
                        num_matches=int(matches.query_points.shape[0]),
                        valid_depth_matches=int(object_points.shape[0]),
                        pnp_inliers=pnp_result.inlier_count,
                        reprojection_error=pnp_result.reprojection_error,
                    ),
                    start,
                )
            if not self._is_pose_delta_reasonable(initial_pose_c2w, pnp_result.pose_c2w):
                return self._log_result(
                    GsplatRefinementResult(
                        None,
                        False,
                        "pose_delta_too_large",
                        num_matches=int(matches.query_points.shape[0]),
                        valid_depth_matches=int(object_points.shape[0]),
                        pnp_inliers=pnp_result.inlier_count,
                        reprojection_error=pnp_result.reprojection_error,
                    ),
                    start,
                )

            result = GsplatRefinementResult(
                pnp_result.pose_c2w,
                True,
                "ok",
                num_matches=int(matches.query_points.shape[0]),
                valid_depth_matches=int(object_points.shape[0]),
                pnp_inliers=pnp_result.inlier_count,
                reprojection_error=pnp_result.reprojection_error,
            )
            if self.config.save_render:
                try:
                    refined_render = self.renderer.render(
                        gaussian_path,
                        pnp_result.pose_c2w,
                        render_camera,
                    )
                    self._save_render_image(robot_id, "render_fix", refined_render.rgb)
                except Exception:
                    logging.exception("Failed to render the accepted refined pose.")
            return self._log_result(result, start)
        except Exception:
            logging.exception("Gsplat refinement failed.")
            return self._log_result(
                GsplatRefinementResult(None, False, "exception"),
                start,
            )

    def _save_render_image(
        self,
        robot_id: str,
        stage: str,
        render_rgb: np.ndarray,
    ) -> None:
        """Save one render without changing the outcome of pose refinement."""
        try:
            output_dir = Path("data/outputs/3dgs_renders")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"id{robot_id}_{stage}.png"
            render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), render_bgr):
                logging.warning("Failed to save 3DGS render to %s", output_path)
        except Exception:
            # Diagnostics must never cause an otherwise valid pose refinement to fail.
            logging.exception("Failed to save 3DGS render.")

    def _build_2d3d_correspondences(
        self,
        render_points: np.ndarray,
        query_points: np.ndarray,
        depth: np.ndarray,
        alpha: np.ndarray,
        pose_c2w: np.ndarray,
        camera: GaussianCamera,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth.shape[:2]
        rounded = np.rint(render_points).astype(np.int32)
        u = rounded[:, 0]
        v = rounded[:, 1]
        valid = (
            (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        z = np.zeros((render_points.shape[0],), dtype=np.float32)
        a = np.zeros((render_points.shape[0],), dtype=np.float32)
        z[valid] = depth[v[valid], u[valid]]
        a[valid] = alpha[v[valid], u[valid]]
        valid &= np.isfinite(z)
        valid &= z >= self.config.min_depth
        valid &= z <= self.config.max_depth
        valid &= a >= self.config.alpha_threshold
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        render_valid = render_points[valid].astype(np.float32)
        query_valid = query_points[valid].astype(np.float32)
        z_valid = z[valid].astype(np.float32)
        x_cam = (render_valid[:, 0] - camera.cx) / camera.fx * z_valid
        y_cam = (render_valid[:, 1] - camera.cy) / camera.fy * z_valid
        points_cam = np.stack([x_cam, y_cam, z_valid, np.ones_like(z_valid)], axis=1)
        points_world = (np.asarray(pose_c2w, dtype=np.float32) @ points_cam.T).T[:, :3]
        return points_world.astype(np.float32), query_valid

    def _is_pose_delta_reasonable(self, old_pose: np.ndarray, new_pose: np.ndarray) -> bool:
        translation_delta = float(np.linalg.norm(new_pose[:2, 3] - old_pose[:2, 3]))
        if translation_delta > self.config.max_translation_delta_m:
            return False
        yaw_delta = abs(np.degrees(self._yaw_delta(old_pose, new_pose)))
        return yaw_delta <= self.config.max_yaw_delta_deg

    @staticmethod
    def _yaw_delta(old_pose: np.ndarray, new_pose: np.ndarray) -> float:
        yaw_old = np.arctan2(old_pose[1, 0], old_pose[0, 0])
        yaw_new = np.arctan2(new_pose[1, 0], new_pose[0, 0])
        return float(np.arctan2(np.sin(yaw_new - yaw_old), np.cos(yaw_new - yaw_old)))

    @staticmethod
    def _log_result(
        result: GsplatRefinementResult,
        start: float,
    ) -> GsplatRefinementResult:
        logging.info(
            "gsplat_refinement result accepted=%s reason=%s matches=%d valid_depth=%d "
            "pnp_inliers=%d reprojection_error=%s time=%.6fs",
            result.accepted,
            result.reason,
            result.num_matches,
            result.valid_depth_matches,
            result.pnp_inliers,
            f"{result.reprojection_error:.6f}" if result.reprojection_error is not None else None,
            time.time() - start,
        )
        return result
