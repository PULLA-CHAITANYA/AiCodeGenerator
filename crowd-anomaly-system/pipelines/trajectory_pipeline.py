"""Detection->tracking->trajectory anomaly pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from models.trajectory.anomaly_detector import TrajectoryAnomalyDetector
from models.trajectory.deepsort_tracker import CrowdTracker
from models.trajectory.trajectory_features import TrajectoryFeatureExtractor
from models.trajectory.yolo_detector import YoloPersonDetector


class TrajectoryPipeline:
    def __init__(self, model_path: str | None = None):
        self.detector = YoloPersonDetector()
        self.tracker = CrowdTracker()
        self.extractor = TrajectoryFeatureExtractor()
        self.anomaly = TrajectoryAnomalyDetector()
        if model_path and Path(model_path).exists():
            self.anomaly = TrajectoryAnomalyDetector.load(model_path)

    def fit_from_video(self, video_path: str | Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        X = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets, frame)
            feats = []
            for tr in tracks:
                feat = self.extractor.update_track(tr.track_id, tr.bbox)
                if feat:
                    feats.append(feat)
            if feats:
                X.append(self.extractor.to_array(feats))
        cap.release()
        if not X:
            raise RuntimeError("No trajectory features extracted")
        self.anomaly.fit(np.vstack(X))

    def run_video(self, video_path: str | Path) -> Dict[str, List[float]]:
        cap = cv2.VideoCapture(str(video_path))
        frame_scores = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets, frame)
            feats = []
            for tr in tracks:
                feat = self.extractor.update_track(tr.track_id, tr.bbox)
                if feat:
                    feats.append(feat)
            if feats and self.anomaly.fitted:
                s = self.anomaly.score(self.extractor.to_array(feats))
                frame_scores.append(float(s.mean()) if len(s) else 0.0)
            else:
                frame_scores.append(0.0)
        cap.release()
        return {"scores": frame_scores}
