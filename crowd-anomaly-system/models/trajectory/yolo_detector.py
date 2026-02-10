"""YOLOv8 people detector wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    cls: int


class YoloPersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.3):
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO detection")
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        out = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection((x1, y1, x2, y2), float(box.conf.item()), int(box.cls.item())))
        return out
