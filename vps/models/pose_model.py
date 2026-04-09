from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from vps.models.pose_model_contract import BasePoseModel, PoseModelOutput
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

        raise NotImplementedError(
            f"Unsupported pose.method for service startup: {self.method}. "
            "The new service stack currently supports only 'vggt' and 'vggt_nav'."
        )

    def infer(
        self,
        query_image: Path,
        ref_images: List[Path],
        depth_paths: Optional[List[Path]] = None,
        poses_paths: Optional[List[Path]] = None,
        k_paths: Optional[List[Path]] = None,
    ) -> PoseModelOutput:
        return self.backend.infer(
            query_image=query_image,
            ref_images=ref_images,
            depth_paths=depth_paths,
            poses_paths=poses_paths,
            k_paths=k_paths,
        )
