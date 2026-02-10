"""Visualization and alert helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="darkgrid")


def overlay_heatmap(frame: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)


def draw_alert(frame: np.ndarray, score: float, threshold: float) -> np.ndarray:
    color = (0, 0, 255) if score >= threshold else (0, 255, 0)
    text = f"Anomaly Score: {score:.3f}"
    cv2.rectangle(frame, (15, 15), (400, 60), color, -1)
    cv2.putText(frame, text, (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def plot_scores(scores: Dict[str, List[float]], out_path: str | Path) -> None:
    plt.figure(figsize=(12, 5))
    for name, vals in scores.items():
        plt.plot(vals, label=name)
    plt.title("Anomaly Score Timeline")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_alert_log(alerts: Iterable[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in alerts:
            f.write(f"{item}\n")
