from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass
class TrajectoryFilterConfig:
    enabled: bool = True
    max_speed_mps: float = 2.0
    speed_margin_m: float = 0.5
    min_time_delta_seconds: float = 0.1
    max_yaw_rate_degps: Optional[float] = None
    yaw_margin_deg: float = 30.0
    reject_invalid_pose: bool = True
    temporal_min_translation_m: float = 0.15
    temporal_min_yaw_deg: float = 8.0
    temporal_min_interval_seconds: float = 0.5


@dataclass
class PoseFilterDecision:
    accepted: bool
    reason: str
    dt_seconds: Optional[float] = None
    distance_m: Optional[float] = None
    speed_mps: Optional[float] = None
    yaw_delta_deg: Optional[float] = None
    yaw_rate_degps: Optional[float] = None


@dataclass
class TemporalFrameDecision:
    should_add: bool
    reason: str
    dt_seconds: Optional[float] = None
    distance_m: Optional[float] = None
    yaw_delta_deg: Optional[float] = None


class TrajectoryFilter:
    """Runtime trajectory sanity checks and temporal-frame sampling policy."""

    def __init__(self, config: Optional[TrajectoryFilterConfig] = None):
        self.config = config or TrajectoryFilterConfig()

    def evaluate_pose(
        self,
        previous_pose: Optional[np.ndarray],
        previous_time: Optional[datetime],
        current_pose: np.ndarray,
        now: datetime,
    ) -> PoseFilterDecision:
        if not self.config.enabled:
            return PoseFilterDecision(accepted=True, reason="disabled")

        if not self._is_valid_pose(current_pose):
            return PoseFilterDecision(
                accepted=not self.config.reject_invalid_pose,
                reason="invalid_pose",
            )

        if previous_pose is None or previous_time is None:
            return PoseFilterDecision(accepted=True, reason="first_pose")

        if not self._is_valid_pose(previous_pose):
            return PoseFilterDecision(accepted=True, reason="previous_pose_invalid")

        dt = self._time_delta_seconds(now, previous_time)
        if dt is None or dt < 0:
            return PoseFilterDecision(accepted=True, reason="invalid_time_delta")

        effective_dt = max(dt, self.config.min_time_delta_seconds)
        distance = self.translation_distance(previous_pose, current_pose)
        max_distance = self.config.max_speed_mps * effective_dt + self.config.speed_margin_m
        if distance > max_distance:
            speed = distance / effective_dt
            return PoseFilterDecision(
                accepted=False,
                reason="speed_jump",
                dt_seconds=dt,
                distance_m=distance,
                speed_mps=speed,
                yaw_delta_deg=self.yaw_delta_deg(previous_pose, current_pose),
            )

        yaw_delta = self.yaw_delta_deg(previous_pose, current_pose)
        yaw_rate = None
        if yaw_delta is not None:
            yaw_rate = yaw_delta / effective_dt
        if (
            self.config.max_yaw_rate_degps is not None
            and yaw_rate is not None
            and yaw_delta > self.config.max_yaw_rate_degps * effective_dt + self.config.yaw_margin_deg
        ):
            return PoseFilterDecision(
                accepted=False,
                reason="yaw_jump",
                dt_seconds=dt,
                distance_m=distance,
                speed_mps=distance / effective_dt,
                yaw_delta_deg=yaw_delta,
                yaw_rate_degps=yaw_rate,
            )

        return PoseFilterDecision(
            accepted=True,
            reason="ok",
            dt_seconds=dt,
            distance_m=distance,
            speed_mps=distance / effective_dt,
            yaw_delta_deg=yaw_delta,
            yaw_rate_degps=yaw_rate,
        )

    def should_add_temporal_frame(
        self,
        previous_temporal_pose: Optional[np.ndarray],
        previous_temporal_time: Optional[datetime],
        current_pose: np.ndarray,
        now: datetime,
    ) -> TemporalFrameDecision:
        if not self.config.enabled:
            return TemporalFrameDecision(should_add=True, reason="disabled")

        if not self._is_valid_pose(current_pose):
            return TemporalFrameDecision(should_add=False, reason="invalid_pose")

        if previous_temporal_pose is None or previous_temporal_time is None:
            return TemporalFrameDecision(should_add=True, reason="first_temporal_frame")

        dt = self._time_delta_seconds(now, previous_temporal_time)
        if dt is None or dt < 0:
            return TemporalFrameDecision(should_add=True, reason="invalid_time_delta")

        distance = self.translation_distance(previous_temporal_pose, current_pose)
        yaw_delta = self.yaw_delta_deg(previous_temporal_pose, current_pose)
        moved_enough = distance >= self.config.temporal_min_translation_m
        rotated_enough = (
            yaw_delta is not None and yaw_delta >= self.config.temporal_min_yaw_deg
        )
        waited_enough = dt >= self.config.temporal_min_interval_seconds

        if moved_enough or rotated_enough:
            return TemporalFrameDecision(
                should_add=True,
                reason="motion_diverse",
                dt_seconds=dt,
                distance_m=distance,
                yaw_delta_deg=yaw_delta,
            )

        if waited_enough and self.config.temporal_min_translation_m <= 0 and self.config.temporal_min_yaw_deg <= 0:
            return TemporalFrameDecision(
                should_add=True,
                reason="time_interval",
                dt_seconds=dt,
                distance_m=distance,
                yaw_delta_deg=yaw_delta,
            )

        return TemporalFrameDecision(
            should_add=False,
            reason="too_similar",
            dt_seconds=dt,
            distance_m=distance,
            yaw_delta_deg=yaw_delta,
        )

    @staticmethod
    def translation_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        a = np.asarray(pose_a, dtype=np.float64)
        b = np.asarray(pose_b, dtype=np.float64)
        return float(np.linalg.norm(a[:2, 3] - b[:2, 3]))

    @staticmethod
    def yaw_delta_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> Optional[float]:
        try:
            yaw_a = TrajectoryFilter._yaw_from_pose(pose_a)
            yaw_b = TrajectoryFilter._yaw_from_pose(pose_b)
        except Exception:
            logging.debug("Failed to compute yaw delta.", exc_info=True)
            return None
        delta = math.atan2(math.sin(yaw_b - yaw_a), math.cos(yaw_b - yaw_a))
        return abs(math.degrees(delta))

    @staticmethod
    def _yaw_from_pose(pose: np.ndarray) -> float:
        matrix = np.asarray(pose, dtype=np.float64)
        return float(math.atan2(matrix[1, 0], matrix[0, 0]))

    @staticmethod
    def _is_valid_pose(pose: np.ndarray) -> bool:
        matrix = np.asarray(pose)
        return matrix.shape == (4, 4) and bool(np.isfinite(matrix).all())

    @staticmethod
    def _time_delta_seconds(now: datetime, previous_time: datetime) -> Optional[float]:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if previous_time.tzinfo is None:
            previous_time = previous_time.replace(tzinfo=timezone.utc)
        return (now - previous_time).total_seconds()
