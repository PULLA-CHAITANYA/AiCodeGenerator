"""End-to-end motion anomaly pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from models.optical_flow_gan.inference import MotionAnomalyDetector
from utils.visualization import draw_alert, overlay_heatmap


class MotionPipeline:
    def __init__(self, threshold: float = 0.12):
        self.detector = MotionAnomalyDetector()
        self.threshold = threshold

    def run_video(self, video_path: str | Path) -> tuple[List[float], List[np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        ok, prev = cap.read()
        if not ok:
            cap.release()
            return [], []
        scores, rendered = [], []

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            score, err_map = self.detector.score_frame_pair(prev, frame)
            vis = overlay_heatmap(frame.copy(), err_map)
            vis = draw_alert(vis, score, self.threshold)
            scores.append(score)
            rendered.append(vis)
            prev = frame

        cap.release()
        return scores, rendered
