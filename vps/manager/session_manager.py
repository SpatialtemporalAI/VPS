from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from vps.session.robot_session import RobotSession


@dataclass
class SessionManager:
    _sessions: Dict[str, RobotSession] = field(default_factory=dict)

    def get_or_create(self, robot_id: str) -> RobotSession:
        session = self._sessions.get(robot_id)
        if session is None:
            session = RobotSession(robot_id=robot_id)
            self._sessions[robot_id] = session
        return session

    def bind_map(self, robot_id: str, map_id: str) -> RobotSession:
        session = self.get_or_create(robot_id)
        session.bind_map(map_id)
        return session
