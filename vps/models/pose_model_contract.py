from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class PoseModelOutput:
    extrinsic: np.ndarray
    intrinsic: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    depth_conf: Optional[np.ndarray] = None
    point_map: Optional[np.ndarray] = None
    point_conf: Optional[np.ndarray] = None
    image_paths: List[Path] = field(default_factory=list)
    temporal_poses: List[np.ndarray] = field(default_factory=list)
    # original_coords: Optional[np.ndarray] = None


class BasePoseModel(Protocol):
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
        ...
