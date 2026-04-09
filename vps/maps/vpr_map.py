from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#路径
@dataclass
class VPRMapConfig:
    map_id: str
    root_dir: Path
    rgb_dir: Path
    poses_dir: Path
    ref_descriptors_path: Path
    query_cache_dir: Optional[Path] = None

#特征数据
@dataclass
class VPRMap:
    """Static VPR database for a single map instance.

    This object only owns map data and lookup helpers. It does not execute
    retrieval policy or similarity search.
    """

    config: VPRMapConfig
    ref_image_paths: List[Path] = field(default_factory=list)
    ref_descriptors: Optional[np.ndarray] = None
    ref_pose_tensor: Optional[np.ndarray] = None
    name_to_index: Dict[str, int] = field(default_factory=dict)
    index_to_name: List[str] = field(default_factory=list)
    _loaded: bool = False

    def load(
        self,
        ref_image_paths: List[Path],
        ref_descriptors: np.ndarray,
        ref_pose_tensor: np.ndarray,
    ) -> None:
        if len(ref_image_paths) != len(ref_descriptors):
            raise ValueError("ref_image_paths and ref_descriptors length mismatch")
        if len(ref_image_paths) != len(ref_pose_tensor):
            raise ValueError("ref_image_paths and ref_pose_tensor length mismatch")

        self.ref_image_paths = list(ref_image_paths)
        self.ref_descriptors = ref_descriptors
        self.ref_pose_tensor = ref_pose_tensor
        self.index_to_name = [path.name for path in self.ref_image_paths]
        self.name_to_index = {
            name: index for index, name in enumerate(self.index_to_name)
        }
        self._loaded = True

    def unload(self) -> None:
        self.ref_image_paths = []
        self.ref_descriptors = None
        self.ref_pose_tensor = None
        self.name_to_index = {}
        self.index_to_name = []
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_ref_descriptors(self) -> np.ndarray:
        self._require_loaded()
        assert self.ref_descriptors is not None
        return self.ref_descriptors

    def get_ref_pose_tensor(self) -> np.ndarray:
        self._require_loaded()
        assert self.ref_pose_tensor is not None
        return self.ref_pose_tensor

    def get_ref_names(self) -> List[str]:
        self._require_loaded()
        return list(self.index_to_name)

    def get_ref_image_path(self, name: str) -> Path:
        self._require_loaded()
        return self.ref_image_paths[self._get_index(name)]

    def get_ref_pose(self, name: str) -> np.ndarray:
        self._require_loaded()
        assert self.ref_pose_tensor is not None
        return self.ref_pose_tensor[self._get_index(name)]

    def _get_index(self, name: str) -> int:
        try:
            return self.name_to_index[name]
        except KeyError as exc:
            raise KeyError(f"Reference image '{name}' not found in map {self.config.map_id}") from exc

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"VPR map '{self.config.map_id}' is not loaded")
