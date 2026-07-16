from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class NavContext:
    map_path: Optional[Path]
    yaml_path: Optional[Path]


@dataclass
class PoseMapConfig:
    map_id: str
    root_dir: Path
    poses_dir: Path
    depth_dir: Optional[Path] = None
    render_rgb_dir: Optional[Path] = None
    render_depth_dir: Optional[Path] = None
    calibration_dir: Optional[Path] = None
    nav_map_path: Optional[Path] = None
    nav_yaml_path: Optional[Path] = None
    vggt_omega_ref_cache_path: Optional[Path] = None
    gaussian_ply_path: Optional[Path] = None
    gaussian_camera: Optional[Dict[str, Any]] = None
    query_camera: Optional[Dict[str, Any]] = None


@dataclass
class PoseMap:
    """Static pose-localization assets for a single map."""

    config: PoseMapConfig
    ref_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    _loaded: bool = False

    def load(self, ref_poses: Dict[str, np.ndarray]) -> None:
        self.ref_poses = dict(ref_poses)
        self._loaded = True

    def unload(self) -> None:
        self.ref_poses = {}
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_ref_pose(self, image_name: str) -> np.ndarray:
        self._require_loaded()
        try:
            return self.ref_poses[image_name]
        except KeyError as exc:
            raise KeyError(
                f"Reference pose '{image_name}' not found in map {self.config.map_id}"
            ) from exc

    def get_ref_depth_path(self, image_name: str) -> Optional[Path]:
        return self._resolve_optional_file(self.config.depth_dir, image_name, ".npy")

    def get_ref_render_path(self, image_name: str) -> Optional[Path]:
        return self._resolve_optional_file(self.config.render_rgb_dir, image_name, ".png")

    def get_ref_render_depth_path(self, image_name: str) -> Optional[Path]:
        return self._resolve_optional_file(
            self.config.render_depth_dir, image_name, ".npy"
        )

    def get_ref_calibration_path(self, image_name: str) -> Optional[Path]:
        return self._resolve_optional_file(
            self.config.calibration_dir, image_name, ".txt"
        )

    def build_nav_context(self) -> NavContext:
        return NavContext(
            map_path=self.config.nav_map_path,
            yaml_path=self.config.nav_yaml_path,
        )

    def _resolve_optional_file(
        self, base_dir: Optional[Path], image_name: str, suffix: str
    ) -> Optional[Path]:
        if base_dir is None:
            return None
        candidate = base_dir / f"{Path(image_name).stem}{suffix}"
        return candidate if candidate.exists() else None

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"Pose map '{self.config.map_id}' is not loaded")
