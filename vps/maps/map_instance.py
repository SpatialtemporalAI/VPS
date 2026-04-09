from __future__ import annotations

from dataclasses import dataclass

from vps.maps.pose_map import PoseMap
from vps.maps.vpr_map import VPRMap


@dataclass
class MapInstance:
    """Runtime map bundle.

    A map instance owns map-scoped assets only. It should be cheap to swap at
    the session layer while sharing model instances across robots.
    """

    map_id: str
    vpr_map: VPRMap
    pose_map: PoseMap

    def is_loaded(self) -> bool:
        return self.vpr_map.is_loaded() and self.pose_map.is_loaded()
