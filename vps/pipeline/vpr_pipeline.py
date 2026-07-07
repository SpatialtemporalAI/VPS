from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Protocol
import logging
import shutil
import time

import numpy as np
import torch

from vps.maps.vpr_map import VPRMap
from vps.utils.find_similar import (
    find_similar,
    find_similar_vpr_pose,
    get_descriptors,
)


@dataclass
class RefMatch:
    ref_name: str
    score: float


@dataclass
class VPRResult:
    query_image_path: Path
    ref_image_paths: List[Path]
    ref_poses: List[np.ndarray] #4x4 c2w
    ref_scores: List[float]


@dataclass
class VPRPipelineConfig:
    top_k: int = 10
    size_num_matched: int = 1
    similarity_threshold: Optional[float] = None
    use_spatial_filtering: bool = False
    spatial_radius: Optional[float] = None
    use_single_query_fast_path: bool = True


class VPRModelProtocol(Protocol):
    def extract_global_descriptors(self, images, output_path: Path) -> Path:
        ...

    def extract_single_global_descriptor(self, image_path: Path, output_path: Path) -> Path:
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
        run_start = time.time()
        with TemporaryDirectory(prefix="vpr_query_") as tmp_dir:
            temp_start = time.time()
            query_desc_path, pairs_path, prepare_timings = self._prepare_query_files(query_image, Path(tmp_dir))
            prepare_total_time = time.time() - temp_start

            db_start = time.time()
            db_names = vpr_map.get_ref_names()
            db_desc = torch.from_numpy(vpr_map.get_ref_descriptors()).float()
            ref_pose_tensor = torch.from_numpy(vpr_map.get_ref_pose_tensor()).float()
            db_prepare_time = time.time() - db_start

            retrieval_start = time.time()
            if self.config.size_num_matched > 1:
                retrieval_mode = "pose_greedy"
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
                retrieval_mode = "spatial" if self.config.use_spatial_filtering else "standard"
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
            retrieval_time = time.time() - retrieval_start

            parse_start = time.time()
            ref_image_paths: List[Path] = []
            ref_poses: List[np.ndarray] = []
            matched_ref_names: List[str] = []
            matched_ref_scores: List[float] = []
            query_scores = self._compute_query_scores(
                query_desc_path=query_desc_path,
                query_name=query_image.name,
                db_names=db_names,
                db_desc=db_desc,
            )
            if pairs_path.exists():
                for line in pairs_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    ref_name = parts[1]
                    matched_ref_names.append(ref_name)
                    matched_ref_scores.append(query_scores.get(ref_name, float("nan")))
                    ref_image_paths.append(vpr_map.get_ref_image_path(ref_name))
                    ref_poses.append(vpr_map.get_ref_pose(ref_name))
            parse_time = time.time() - parse_start
            total_time = time.time() - run_start
            matched_ref_with_scores = [
                f"{name}:{score:.6f}" if np.isfinite(score) else f"{name}:nan"
                for name, score in zip(matched_ref_names, matched_ref_scores)
            ]

            logging.info(
                "VPR retrieval result: map_id=%s query=%s num_refs=%d ref_names=%s ref_matches=%s",
                vpr_map.config.map_id,
                query_image.name,
                len(matched_ref_names),
                matched_ref_names,
                matched_ref_with_scores,
            )
            logging.info(
                "VPR pipeline detail: map_id=%s query=%s mode=%s db_size=%d top_k=%d "
                "size_num_matched=%d spatial_filter=%s spatial_radius=%s "
                "copy_query=%.6fs extract_query=%.6fs prepare_query_total=%.6fs "
                "extract_mode=%s db_prepare=%.6fs retrieval=%.6fs parse_pairs=%.6fs total=%.6fs "
                "desc_shape=%s pose_shape=%s tmp_dir=%s",
                vpr_map.config.map_id,
                query_image.name,
                retrieval_mode,
                len(db_names),
                self.config.top_k,
                self.config.size_num_matched,
                self.config.use_spatial_filtering,
                self.config.spatial_radius,
                prepare_timings["copy_query"],
                prepare_timings["extract_query"],
                prepare_total_time,
                prepare_timings["extract_mode"],
                db_prepare_time,
                retrieval_time,
                parse_time,
                total_time,
                tuple(db_desc.shape),
                tuple(ref_pose_tensor.shape),
                tmp_dir,
            )

            return VPRResult(
                query_image_path=query_image,
                ref_image_paths=ref_image_paths,
                ref_poses=ref_poses,
                ref_scores=matched_ref_scores,
            )

    def _compute_query_scores(
        self,
        query_desc_path: Path,
        query_name: str,
        db_names: List[str],
        db_desc: torch.Tensor,
    ) -> dict[str, float]:
        query_desc = get_descriptors([query_name], query_desc_path).float()
        device = db_desc.device
        scores = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))[0]
        return {
            name: float(score)
            for name, score in zip(db_names, scores.detach().cpu().tolist())
        }

    def _prepare_query_files(
        self, query_image: Path, tmp_dir: Path
    ) -> tuple[Path, Path, dict[str, float]]:
        query_dir = tmp_dir / "query"
        query_dir.mkdir(parents=True, exist_ok=True)

        copy_start = time.time()
        query_copy_path = query_dir / query_image.name
        shutil.copy2(query_image, query_copy_path)
        copy_time = time.time() - copy_start

        query_desc_path = tmp_dir / "query.h5"
        pairs_path = tmp_dir / "pairs.txt"
        extract_start = time.time()
        extract_single = getattr(self.model, "extract_single_global_descriptor", None)
        if self.config.use_single_query_fast_path and extract_single is not None:
            extract_single(query_copy_path, query_desc_path)
            extract_mode = "single_fast"
        else:
            self.model.extract_global_descriptors(query_dir, query_desc_path)
            extract_mode = "hloc_main"
        extract_time = time.time() - extract_start
        return query_desc_path, pairs_path, {
            "copy_query": copy_time,
            "extract_query": extract_time,
            "extract_mode": extract_mode,
        }
