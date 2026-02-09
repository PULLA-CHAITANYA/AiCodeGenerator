"""Weighted fusion for final crowd anomaly decisions."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.config import CONFIG


def minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class EnsembleAnomalyScorer:
    def __init__(self, weights: Dict[str, float] | None = None, threshold: float | None = None):
        self.weights = weights or CONFIG.ensemble.weights
        self.threshold = threshold if threshold is not None else CONFIG.ensemble.anomaly_threshold

    def fuse(self, motion: List[float], trajectory: List[float], density: List[float]) -> Dict[str, List[float]]:
        n = min(len(motion), len(trajectory), len(density))
        m = minmax(np.array(motion[:n], dtype=np.float32))
        t = minmax(np.array(trajectory[:n], dtype=np.float32))
        d = minmax(np.array(density[:n], dtype=np.float32))

        final = self.weights["motion"] * m + self.weights["trajectory"] * t + self.weights["density"] * d
        flags = (final >= self.threshold).astype(int)
        return {
            "motion_norm": m.tolist(),
            "trajectory_norm": t.tolist(),
            "density_norm": d.tolist(),
            "final": final.tolist(),
            "anomaly": flags.tolist(),
        }
