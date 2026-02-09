"""Train trajectory anomaly detector from normal video."""
from __future__ import annotations

import argparse

from pipelines.trajectory_pipeline import TrajectoryPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Normal behavior video path")
    parser.add_argument("--out", default="checkpoints/trajectory_iforest.joblib")
    args = parser.parse_args()

    pipe = TrajectoryPipeline()
    pipe.fit_from_video(args.video)
    pipe.anomaly.save(args.out)
    print(f"Saved trajectory model to {args.out}")
