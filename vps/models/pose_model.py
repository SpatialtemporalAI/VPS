from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
from vps.models.omnivggt_model import OmniVGGTModel
from vps.models.vggt_omega_model import VGGTOmegaModel
from vps.models.vggt_model import VGGTModel


class PoseModel:
    """Unified pose-model entrypoint configured by ``pose.method``."""

    def __init__(self, config: Dict):
        self.config = config
        self.method = str(config.get("pose", {}).get("method", "vggt")).lower()
        self.backend = self._create_backend()

    def _create_backend(self) -> BasePoseModel:
        # The new models/pipeline stack has only migrated the VGGT-style model so far.
        if self.method in {"vggt", "vggt_nav"}:
            return VGGTModel(self.config)
        if self.method == "omnivggt":
            return OmniVGGTModel(self.config)
        if self.method in {"vggt_omega", "vggt-omega", "vggtomega"}:
            return VGGTOmegaModel(self.config)

        raise NotImplementedError(
            f"Unsupported pose.method for service startup: {self.method}. "
            "The new service stack currently supports 'vggt', 'vggt_nav', 'omnivggt', and 'vggt_omega'."
        )

    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        ref_poses: Optional[List[np.ndarray]] = None,
        depth_paths: Optional[List[Path]] = None,
        poses_paths: Optional[List[Path]] = None,
        k_paths: Optional[List[Path]] = None,
        ref_cache_path: Optional[Path] = None,
        robot_id: Optional[str] = None,
        map_id: Optional[str] = None,
    ) -> PoseModelOutput:
        if self.method in {"vggt_omega", "vggt-omega", "vggtomega"}:
            return self.backend.infer(
                query_image=query_image,
                ref_images=ref_images,
                ref_poses=ref_poses,
                depth_paths=depth_paths,
                poses_paths=poses_paths,
                k_paths=k_paths,
                ref_cache_path=ref_cache_path,
                robot_id=robot_id,
                map_id=map_id,
            )
        return self.backend.infer(
            query_image=query_image,
            ref_images=ref_images,
            ref_poses=ref_poses,
            depth_paths=depth_paths,
            poses_paths=poses_paths,
            k_paths=k_paths,
        )

    def update_temporal_pose(
        self,
        robot_id: str,
        map_id: str,
        query_image: Path,
        pose_c2w: np.ndarray,
    ) -> None:
        update_fn = getattr(self.backend, "update_temporal_pose", None)
        if update_fn is None:
            return
        update_fn(
            robot_id=robot_id,
            map_id=map_id,
            query_image=query_image,
            pose_c2w=pose_c2w,
        )
