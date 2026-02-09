"""Crowd density anomaly pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from models.density.density_anomaly import DensityAnomalyDetector
from models.density.density_estimator import DensityEstimator


class DensityPipeline:
    def __init__(self, detector_path: str | None = None):
        self.estimator = DensityEstimator()
        self.anomaly = DensityAnomalyDetector()
        if detector_path and Path(detector_path).exists():
            self.anomaly = DensityAnomalyDetector.load(detector_path)

    def fit_from_video(self, normal_video_path: str | Path) -> List[float]:
        counts = []
        cap = cv2.VideoCapture(str(normal_video_path))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            _, count = self.estimator.estimate(frame)
            counts.append(count)
        cap.release()
        if not counts:
            raise RuntimeError("No density counts extracted")
        self.anomaly.fit(np.array(counts, dtype=np.float32))
        return counts

    def run_video(self, video_path: str | Path) -> Dict[str, List[float]]:
        cap = cv2.VideoCapture(str(video_path))
        counts, scores = [], []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            _, count = self.estimator.estimate(frame)
            counts.append(count)
            if self.anomaly.fitted:
                scores.append(self.anomaly.score(count))
            else:
                scores.append(0.0)
        cap.release()
        return {"counts": counts, "scores": scores}
