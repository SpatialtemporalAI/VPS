from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PnPResult:
    pose_c2w: np.ndarray | None
    inlier_count: int
    reprojection_error: float | None


class PnPSolver:
    def __init__(
        self,
        ransac_reproj_error: float = 4.0,
        confidence: float = 0.999,
        iterations_count: int = 1000,
    ):
        self.ransac_reproj_error = float(ransac_reproj_error)
        self.confidence = float(confidence)
        self.iterations_count = int(iterations_count)

    def solve(
        self,
        object_points_world: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> PnPResult:
        object_points_world = np.asarray(object_points_world, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        if object_points_world.shape[0] < 4 or image_points.shape[0] < 4:
            return PnPResult(None, 0, None)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_world,
            image_points,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.ransac_reproj_error,
            confidence=self.confidence,
            iterationsCount=self.iterations_count,
        )
        if not ok or inliers is None or len(inliers) == 0:
            return PnPResult(None, 0, None)

        inlier_indices = inliers.reshape(-1)
        if len(inlier_indices) >= 4:
            cv2.solvePnP(
                object_points_world[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                None,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        rotation, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rotation.astype(np.float32)
        w2c[:3, 3] = tvec.reshape(3).astype(np.float32)
        c2w = np.linalg.inv(w2c).astype(np.float32)
        error = self._mean_reprojection_error(
            object_points_world[inlier_indices],
            image_points[inlier_indices],
            rvec,
            tvec,
            camera_matrix,
        )
        return PnPResult(c2w, int(len(inlier_indices)), error)

    @staticmethod
    def _mean_reprojection_error(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
