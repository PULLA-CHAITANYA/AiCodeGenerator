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

    def run_video(self, video_path: str | Path) -> Dict[str, List]:
        cap = cv2.VideoCapture(str(video_path))
        frame_scores: List[float] = []
        frame_tracks: List[List[dict]] = []
        frame_feature_stats: List[dict] = []

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            dets = self.detector.detect(frame)
            tracks = self.tracker.update(dets, frame)
            feats = []
            track_entries: List[dict] = []

            for tr in tracks:
                feat = self.extractor.update_track(tr.track_id, tr.bbox)
                if feat:
                    feats.append(feat)
                track_entries.append({"track_id": tr.track_id, "bbox": list(tr.bbox)})

            if feats:
                f_arr = self.extractor.to_array(feats)
                feature_stat = {
                    "mean_speed": float(np.mean(f_arr[:, 0])),
                    "mean_acceleration": float(np.mean(f_arr[:, 1])),
                    "mean_direction_change": float(np.mean(f_arr[:, 2])),
                    "mean_curvature": float(np.mean(f_arr[:, 3])),
                }
            else:
                feature_stat = {
                    "mean_speed": 0.0,
                    "mean_acceleration": 0.0,
                    "mean_direction_change": 0.0,
                    "mean_curvature": 0.0,
                }

            if feats and self.anomaly.fitted:
                s = self.anomaly.score(self.extractor.to_array(feats))
                frame_scores.append(float(s.mean()) if len(s) else 0.0)
            else:
                frame_scores.append(0.0)

            frame_tracks.append(track_entries)
            frame_feature_stats.append(feature_stat)

        cap.release()
        return {"scores": frame_scores, "tracks": frame_tracks, "feature_stats": frame_feature_stats}
