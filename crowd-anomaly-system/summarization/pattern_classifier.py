"""Deterministic pattern labeling from module-derived features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PatternResult:
    pattern: str
    confidence: float
    anomaly_type: str
    contributors: List[str]


class PatternClassifier:
    """Rule-based classifier driven by measured features (no random text)."""

    def classify(
        self,
        motion_scores: List[float],
        trajectory_scores: List[float],
        density_scores: List[float],
        density_counts: List[float],
        trajectory_features: List[Dict[str, float]],
    ) -> PatternResult:
        m = np.asarray(motion_scores, dtype=np.float32)
        t = np.asarray(trajectory_scores, dtype=np.float32)
        d = np.asarray(density_scores, dtype=np.float32)
        c = np.asarray(density_counts, dtype=np.float32) if density_counts else np.array([0.0], dtype=np.float32)

        motion_spike = float(np.max(np.diff(m, prepend=m[0]))) if len(m) > 1 else 0.0
        density_growth = float((c[-1] - c[0]) / (abs(c[0]) + 1e-6)) if len(c) > 1 else 0.0
        mean_density = float(np.mean(c))

        dir_change = float(np.mean([f.get("mean_direction_change", 0.0) for f in trajectory_features])) if trajectory_features else 0.0
        mean_speed = float(np.mean([f.get("mean_speed", 0.0) for f in trajectory_features])) if trajectory_features else 0.0

        if np.mean(d) > 0.65 and density_growth > 0.2:
            return PatternResult("Overcrowding buildup", min(0.99, np.mean(d) + 0.1), "overcrowding", ["density"])

        if np.mean(m) > 0.7 and motion_spike > 0.18 and np.mean(t) < 0.45:
            return PatternResult("Panic dispersal", min(0.99, np.mean(m)), "panic_movement", ["motion"])

        if np.mean(m) > 0.65 and mean_speed > 5.0 and np.mean(t) > 0.55:
            return PatternResult("Stampede-like motion", min(0.99, (np.mean(m) + np.mean(t)) / 2), "stampede", ["motion", "trajectory"])

        if dir_change > 1.1 and np.mean(t) > 0.5 and np.mean(m) > 0.45:
            return PatternResult("Directional chaos", min(0.99, (np.mean(t) + np.mean(m)) / 2), "directional_chaos", ["motion", "trajectory"])

        if mean_speed > 4.5 and np.mean(t) > 0.5:
            return PatternResult("Sudden acceleration", min(0.99, np.mean(t)), "acceleration_anomaly", ["trajectory"])

        if mean_density < np.percentile(c, 70) and np.mean(m) < 0.45 and np.mean(t) < 0.45:
            return PatternResult("Normal flow", 0.85, "normal", ["motion", "trajectory", "density"])

        return PatternResult("Normal flow", 0.65, "normal", ["motion", "trajectory", "density"])
