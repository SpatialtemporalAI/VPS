from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RobotSession:
    robot_id: str
    active_map_id: Optional[str] = None
    last_pose: Optional[np.ndarray] = None
    last_update_time: Optional[datetime] = None
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)

    def bind_map(self, map_id: str) -> None:
        self.active_map_id = map_id

    def update_pose(self, pose: np.ndarray, update_time: Optional[datetime] = None) -> None:
        self.last_pose = pose
        self.last_update_time = update_time or datetime.now(timezone.utc)
