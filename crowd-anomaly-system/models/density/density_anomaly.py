"""Time-series anomaly detection for crowd density."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class DensityAnomalyDetector:
    def __init__(self, contamination: float = 0.07):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False

    def fit(self, counts: np.ndarray) -> None:
        X = counts.reshape(-1, 1)
        self.model.fit(X)
        self.fitted = True

    def score(self, count: float) -> float:
        if not self.fitted:
            raise RuntimeError("Density anomaly detector must be fit before scoring")
        raw = -self.model.decision_function(np.array([[count]]))[0]
        return float(1 / (1 + np.exp(-raw)))

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "fitted": self.fitted}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        inst = cls()
        inst.model = data["model"]
        inst.fitted = data["fitted"]
        return inst
