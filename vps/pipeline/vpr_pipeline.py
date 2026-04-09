from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Protocol
import shutil

import numpy as np
import torch

from vps.maps.vpr_map import VPRMap
from vps.utils.find_similar import (
    find_similar,
    find_similar_vpr_pose,
)


@dataclass
class RefMatch:
    ref_name: str
    score: float


@dataclass
class VPRResult:
    query_image_path: Path
    ref_image_paths: List[Path]


@dataclass
class VPRPipelineConfig:
    top_k: int = 10
    size_num_matched: int = 1
    similarity_threshold: Optional[float] = None
    use_spatial_filtering: bool = False
    spatial_radius: Optional[float] = None


class VPRModelProtocol(Protocol):
    def extract_global_descriptors(self, images, output_path: Path) -> Path:
        ...


class VPRPipeline:
    """Coordinate VPR retrieval.

    Retrieval policy lives here (pipeline layer), not in model/map classes.
    """

    def __init__(self, model: VPRModelProtocol, config: VPRPipelineConfig):
        self.model = model
        self.config = config

    def run(
        self,
        query_image: Path,
        vpr_map: VPRMap,
        last_pose: Optional[np.ndarray] = None,
    ) -> VPRResult:
        if not vpr_map.is_loaded():
            raise RuntimeError(
                f"VPR map '{vpr_map.config.map_id}' is not prepared. "
                "Call MapManager.prepare_map(...) before running localization."
            )
        with TemporaryDirectory(prefix="vpr_query_") as tmp_dir:
            query_desc_path, pairs_path = self._prepare_query_files(query_image, Path(tmp_dir))
            db_names = vpr_map.get_ref_names()
            db_desc = torch.from_numpy(vpr_map.get_ref_descriptors()).float()
            ref_pose_tensor = torch.from_numpy(vpr_map.get_ref_pose_tensor()).float()

            if self.config.size_num_matched > 1:
                find_similar_vpr_pose(
                    query_name=query_image.name,
                    query_descriptors=query_desc_path,
                    db_descriptors=[vpr_map.config.ref_descriptors_path],
                    db_names=db_names,
                    db_desc=db_desc,
                    output=pairs_path,
                    num_matched=self.config.top_k,
                    size_num_matched=self.config.size_num_matched,
                    ref_poses_tensor=ref_pose_tensor,
                )
            else:
                if not self.config.use_spatial_filtering:
                    last_pose = None
                find_similar(
                    query_descriptors=query_desc_path,
                    db_descriptors=[vpr_map.config.ref_descriptors_path],
                    db_names=db_names,
                    db_desc=db_desc,
                    output=pairs_path,
                    num_matched=self.config.top_k,
                    similarity_threshold=self.config.similarity_threshold,
                    last_pose=last_pose,
                    spatial_radius=self.config.spatial_radius,
                    use_spatial_filtering=self.config.use_spatial_filtering,
                    ref_poses_tensor=ref_pose_tensor,
                )

            ref_image_paths: List[Path] = []
            if pairs_path.exists():
                for line in pairs_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    ref_name = parts[1]
                    ref_image_paths.append(vpr_map.get_ref_image_path(ref_name))

            return VPRResult(
                query_image_path=query_image,
                ref_image_paths=ref_image_paths,
            )

    def _prepare_query_files(
        self, query_image: Path, tmp_dir: Path
    ) -> tuple[Path, Path]:
        query_dir = tmp_dir / "query"
        query_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(query_image, query_dir / query_image.name)
        query_desc_path = tmp_dir / "query.h5"
        pairs_path = tmp_dir / "pairs.txt"
        self.model.extract_global_descriptors(query_dir, query_desc_path)
        return query_desc_path, pairs_path
