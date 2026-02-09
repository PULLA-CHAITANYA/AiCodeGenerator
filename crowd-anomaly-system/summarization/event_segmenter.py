"""Temporal event segmentation from fused anomaly score timelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EventSegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    severity: str
    peak_score: float
    mean_score: float


class EventSegmenter:
    def __init__(self, warning_threshold: float = 0.45, critical_threshold: float = 0.7, delta_threshold: float = 0.12):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.delta_threshold = delta_threshold

    def _score_to_severity(self, score: float) -> str:
        if score >= self.critical_threshold:
            return "critical"
        if score >= self.warning_threshold:
            return "warning"
        return "normal"

    def segment(self, final_scores: List[float], fps: float) -> List[EventSegment]:
        if not final_scores:
            return []

        scores = np.asarray(final_scores, dtype=np.float32)
        deltas = np.abs(np.diff(scores, prepend=scores[0]))

        segments: List[EventSegment] = []
        start_idx = 0
        current_severity = self._score_to_severity(float(scores[0]))

        for idx in range(1, len(scores)):
            severity = self._score_to_severity(float(scores[idx]))
            significant_change = deltas[idx] >= self.delta_threshold
            if severity != current_severity or significant_change:
                seg_scores = scores[start_idx:idx]
                segments.append(
                    EventSegment(
                        start_frame=start_idx,
                        end_frame=idx - 1,
                        start_time=start_idx / fps,
                        end_time=(idx - 1) / fps,
                        severity=current_severity,
                        peak_score=float(np.max(seg_scores)),
                        mean_score=float(np.mean(seg_scores)),
                    )
                )
                start_idx = idx
                current_severity = severity

        seg_scores = scores[start_idx:]
        segments.append(
            EventSegment(
                start_frame=start_idx,
                end_frame=len(scores) - 1,
                start_time=start_idx / fps,
                end_time=(len(scores) - 1) / fps,
                severity=current_severity,
                peak_score=float(np.max(seg_scores)),
                mean_score=float(np.mean(seg_scores)),
            )
        )
        return segments
