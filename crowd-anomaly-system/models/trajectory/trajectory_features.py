"""Feature extraction from tracked trajectories."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np


@dataclass
class TrajectoryFeature:
    track_id: int
    speed: float
    acceleration: float
    direction_change: float
    curvature: float


class TrajectoryFeatureExtractor:
    def __init__(self, history: int = 64):
        self.positions: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=history))
        self.prev_speed: Dict[int, float] = defaultdict(float)

    def update_track(self, track_id: int, bbox: tuple[int, int, int, int]) -> TrajectoryFeature | None:
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        hist = self.positions[track_id]
        hist.append(center)
        if len(hist) < 3:
            return None

        p1, p2, p3 = np.array(hist[-3]), np.array(hist[-2]), np.array(hist[-1])
        v1 = p2 - p1
        v2 = p3 - p2
        speed = float(np.linalg.norm(v2))
        acceleration = speed - self.prev_speed[track_id]
        self.prev_speed[track_id] = speed

        def angle_between(a: np.ndarray, b: np.ndarray) -> float:
            na = np.linalg.norm(a) + 1e-8
            nb = np.linalg.norm(b) + 1e-8
            cos_val = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            return float(np.arccos(cos_val))

        direction_change = angle_between(v1, v2)
        curvature = direction_change / (np.linalg.norm(v2) + 1e-6)

        return TrajectoryFeature(
            track_id=track_id,
            speed=speed,
            acceleration=float(acceleration),
            direction_change=direction_change,
            curvature=float(curvature),
        )

    @staticmethod
    def to_array(features: List[TrajectoryFeature]) -> np.ndarray:
        if not features:
            return np.empty((0, 4), dtype=np.float32)
        return np.array(
            [[f.speed, f.acceleration, f.direction_change, f.curvature] for f in features],
            dtype=np.float32,
        )
