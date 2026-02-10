"""Generate multiple synthetic crowd videos (normal + anomaly) with frame-level labels.

This script is intended for quick local benchmarking when public datasets are unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _render_video(
    out_path: Path,
    labels_out: Path,
    n_frames: int,
    fps: int,
    crowd_size: int,
    anomaly_ranges: list[tuple[int, int]],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    w, h = 640, 360
    pos = rng.uniform([20, 20], [w - 20, h - 20], size=(crowd_size, 2))
    vel = rng.normal(0, 0.8, size=(crowd_size, 2))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_labels: list[int] = []

    for t in range(n_frames):
        is_anom = any(start <= t <= end for start, end in anomaly_ranges)
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if is_anom:
            shock = rng.normal(0, 2.8, size=(crowd_size, 2))
            vel += shock
            if (t // 10) % 2 == 0:
                vel[:, 0] *= -1.0  # directional chaos
        else:
            vel += rng.normal(0, 0.18, size=(crowd_size, 2))

        vel = np.clip(vel, -5.0, 5.0)
        pos += vel
        pos[:, 0] = np.clip(pos[:, 0], 5, w - 5)
        pos[:, 1] = np.clip(pos[:, 1], 5, h - 5)

        for i in range(crowd_size):
            color = (0, 180, 255) if is_anom else (0, 220, 120)
            cv2.circle(frame, (int(pos[i, 0]), int(pos[i, 1])), 3, color, -1)

        tag = "ANOMALY" if is_anom else "NORMAL"
        cv2.putText(frame, f"GT: {tag}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        frame_labels.append(1 if is_anom else 0)

    writer.release()

    labels_out.write_text(
        json.dumps(
            {
                "video": out_path.name,
                "fps": fps,
                "anomaly_ranges": anomaly_ranges,
                "frame_labels": frame_labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def generate(dataset_dir: Path, n_normal: int = 3, n_anomaly: int = 3) -> None:
    videos_dir = dataset_dir / "videos"
    labels_dir = dataset_dir / "labels"
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    fps = 20
    n_frames = 220
    for i in range(n_normal):
        out = videos_dir / f"normal_{i:02d}.mp4"
        lab = labels_dir / f"normal_{i:02d}.json"
        _render_video(out, lab, n_frames, fps, crowd_size=90, anomaly_ranges=[], seed=100 + i)

    anomaly_templates = [
        [(60, 95), (150, 185)],
        [(40, 80), (130, 170)],
        [(70, 120)],
    ]
    for i in range(n_anomaly):
        out = videos_dir / f"anomaly_{i:02d}.mp4"
        lab = labels_dir / f"anomaly_{i:02d}.json"
        ranges = anomaly_templates[i % len(anomaly_templates)]
        _render_video(out, lab, n_frames, fps, crowd_size=95, anomaly_ranges=ranges, seed=300 + i)

    print(f"Synthetic dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/synth_benchmark"))
    parser.add_argument("--n_normal", type=int, default=3)
    parser.add_argument("--n_anomaly", type=int, default=3)
    args = parser.parse_args()
    generate(args.dataset_dir, args.n_normal, args.n_anomaly)
