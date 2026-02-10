"""Trajectory anomaly model using Isolation Forest."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class TrajectoryAnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        if len(X) == 0:
            raise ValueError("Cannot fit trajectory detector with empty features")
        self.model.fit(X)
        self.fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        if not self.fitted:
            raise RuntimeError("Trajectory anomaly detector must be fit before scoring")
        raw = -self.model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        inst = cls()
        inst.model = data["model"]
        inst.fitted = data["fitted"]
        return inst
