"""Evaluate full anomaly pipeline against frame-level labels on multiple videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from run import run_full_pipeline


def _load_gt(label_path: Path) -> np.ndarray:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    return np.asarray(data["frame_labels"], dtype=np.int32)


def evaluate(dataset_dir: Path, out_json: Path) -> dict:
    videos_dir = dataset_dir / "videos"
    labels_dir = dataset_dir / "labels"

    all_gt: list[int] = []
    all_pred: list[int] = []
    video_rows = []

    for video in sorted(videos_dir.glob("*.mp4")):
        label_file = labels_dir / f"{video.stem}.json"
        if not label_file.exists():
            print(f"Skipping {video.name}: missing {label_file.name}")
            continue

        gt = _load_gt(label_file)
        result = run_full_pipeline(str(video))
        pred = np.asarray(result["flags"], dtype=np.int32)
        n = min(len(gt), len(pred))
        gt = gt[:n]
        pred = pred[:n]

        row = {
            "video": video.name,
            "frames_evaluated": int(n),
            "accuracy": float(accuracy_score(gt, pred)),
            "precision": float(precision_score(gt, pred, zero_division=0)),
            "recall": float(recall_score(gt, pred, zero_division=0)),
            "f1": float(f1_score(gt, pred, zero_division=0)),
            "mean_motion_score": float(np.mean(result["scores"]["motion"][:n])) if n else 0.0,
            "mean_trajectory_score": float(np.mean(result["scores"]["trajectory"][:n])) if n else 0.0,
            "mean_density_score": float(np.mean(result["scores"]["density"][:n])) if n else 0.0,
            "mean_fused_score": float(np.mean(result["scores"]["final"][:n])) if n else 0.0,
        }
        video_rows.append(row)

        all_gt.extend(gt.tolist())
        all_pred.extend(pred.tolist())
        print(f"{video.name}: acc={row['accuracy']:.3f} f1={row['f1']:.3f}")

    overall = {
        "accuracy": float(accuracy_score(all_gt, all_pred)) if all_gt else 0.0,
        "precision": float(precision_score(all_gt, all_pred, zero_division=0)) if all_gt else 0.0,
        "recall": float(recall_score(all_gt, all_pred, zero_division=0)) if all_gt else 0.0,
        "f1": float(f1_score(all_gt, all_pred, zero_division=0)) if all_gt else 0.0,
        "total_frames": int(len(all_gt)),
        "videos": len(video_rows),
    }

    report = {"overall": overall, "per_video": video_rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Overall:", overall)
    print(f"Saved evaluation report: {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/synth_benchmark"))
    parser.add_argument("--out_json", type=Path, default=Path("outputs/evaluation_report.json"))
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.out_json)
