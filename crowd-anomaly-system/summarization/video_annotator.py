"""Generate real-time readable annotated video with overlays and labels."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils.flow_utils import compute_farneback_flow, flow_to_mag_angle


class VideoAnnotator:
    def __init__(self):
        self.tracks_history: Dict[int, List[tuple[int, int]]] = {}

    def _draw_trajectories(self, frame: np.ndarray, tracks: List[dict]) -> None:
        for tr in tracks:
            track_id = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.tracks_history.setdefault(track_id, []).append((cx, cy))
            self.tracks_history[track_id] = self.tracks_history[track_id][-20:]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 200, 30), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 200, 30), 1)

            pts = self.tracks_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 200, 0), 2)

    @staticmethod
    def _overlay_density(frame: np.ndarray, density_score: float) -> None:
        bar_w = int(min(220, max(0, density_score * 220)))
        cv2.rectangle(frame, (20, 78), (245, 100), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 78), (20 + bar_w, 100), (255, 80, 80), -1)
        cv2.putText(frame, f"Density:{density_score:.2f}", (22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    @staticmethod
    def _overlay_flow_heatmap(frame: np.ndarray, prev_frame: np.ndarray | None) -> np.ndarray:
        if prev_frame is None:
            return frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = compute_farneback_flow(prev_gray, curr_gray)
        mag, _ = flow_to_mag_angle(flow)
        flow_map = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(flow_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.72, heat, 0.28, 0)

    def annotate(
        self,
        video_path: str | Path,
        output_path: str | Path,
        fused_scores: List[float],
        severity_labels: List[str],
        pattern_labels: List[str],
        trajectory_tracks: List[List[dict]],
        density_scores: List[float],
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = 0
        prev = None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame_idx >= len(fused_scores):
                break

            composed = self._overlay_flow_heatmap(frame, prev)
            tracks = trajectory_tracks[frame_idx] if frame_idx < len(trajectory_tracks) else []
            self._draw_trajectories(composed, tracks)

            density_score = density_scores[frame_idx] if frame_idx < len(density_scores) else 0.0
            self._overlay_density(composed, density_score)

            sev = severity_labels[frame_idx]
            pattern = pattern_labels[frame_idx]
            score = fused_scores[frame_idx]
            color = (0, 255, 0) if sev == "normal" else (0, 215, 255) if sev == "warning" else (0, 0, 255)

            cv2.rectangle(composed, (15, 15), (520, 66), color, -1)
            cv2.putText(composed, f"Severity: {sev.upper()}  Score: {score:.2f}", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(composed, f"Pattern: {pattern}", (22, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

            writer.write(composed)
            prev = frame.copy()
            frame_idx += 1

        writer.release()
        cap.release()
