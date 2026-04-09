from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from typing import Protocol
import threading

import numpy as np
from hloc.utils.io import list_h5_names

from vps.utils.find_similar import get_descriptors

from vps.maps.map_instance import MapInstance


class VPRModelProtocol(Protocol):
    def extract_global_descriptors(self, images, output_path: Path) -> Path:
        ...


@dataclass
class MapManager:
    """Registry for multiple runtime map instances."""

    _maps: Dict[str, MapInstance] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _prepare_locks: Dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)

    def register(self, map_instance: MapInstance) -> None:
        with self._lock:
            self._maps[map_instance.map_id] = map_instance
            self._prepare_locks.setdefault(map_instance.map_id, threading.Lock())

    def get(self, map_id: str) -> MapInstance:
        with self._lock:
            try:
                return self._maps[map_id]
            except KeyError as exc:
                raise KeyError(f"Map '{map_id}' is not registered") from exc

    def is_loaded(self, map_id: str) -> bool:
        return self.get(map_id).is_loaded()

    def prepare_map(
        self,
        map_id: str,
        vpr_model: VPRModelProtocol,
        check_ref: bool = False,
    ) -> None:
        with self._lock:
            map_instance = self.get(map_id)
            prepare_lock = self._prepare_locks.setdefault(map_id, threading.Lock())

        # Only one thread can prepare the same map at a time.
        with prepare_lock:
            self._prepare_map_locked(map_instance, vpr_model, check_ref)

    def _prepare_map_locked(
        self,
        map_instance: MapInstance,
        vpr_model: VPRModelProtocol,
        check_ref: bool,
    ) -> None:
        vpr_map = map_instance.vpr_map
        descriptor_path = vpr_map.config.ref_descriptors_path
        should_rebuild = check_ref or not descriptor_path.exists()
        if should_rebuild:
            vpr_model.extract_global_descriptors(vpr_map.config.rgb_dir, descriptor_path)

        db_names = list_h5_names(descriptor_path)
        if not db_names:
            raise RuntimeError(f"No descriptors found in {descriptor_path}")

        db_desc = get_descriptors(db_names, descriptor_path).cpu().numpy()
        ref_image_paths = [vpr_map.config.rgb_dir / name for name in db_names]
        ref_poses = []
        for name in db_names:
            pose_path = vpr_map.config.poses_dir / f"{Path(name).stem}.txt"
            if not pose_path.exists():
                raise FileNotFoundError(f"Missing pose file: {pose_path}")
            ref_poses.append(np.loadtxt(pose_path).reshape(4, 4))

        vpr_map.load(
            ref_image_paths=ref_image_paths,
            ref_descriptors=db_desc,
            ref_pose_tensor=np.stack(ref_poses, axis=0),
        )
