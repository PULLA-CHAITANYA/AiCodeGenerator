"""DeepSORT-based multi-object tracker wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:  # pragma: no cover
    DeepSort = None


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confirmed: bool = True


class CrowdTracker:
    def __init__(self, max_age: int = 30):
        if DeepSort is None:
            raise ImportError("deep-sort-realtime is required for DeepSORT tracking")
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame) -> List[Track]:
        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            ds_inputs.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, "person"))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
        out = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            out.append(Track(track_id=tr.track_id, bbox=(int(l), int(t), int(r), int(b)), confirmed=True))
        return out
