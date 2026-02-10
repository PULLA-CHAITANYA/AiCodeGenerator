"""Train density anomaly model from normal video."""
from __future__ import annotations

import argparse

from pipelines.density_pipeline import DensityPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Normal behavior video path")
    parser.add_argument("--out", default="checkpoints/density_iforest.joblib")
    args = parser.parse_args()

    pipe = DensityPipeline()
    counts = pipe.fit_from_video(args.video)
    pipe.anomaly.save(args.out)
    print(f"Saved density model to {args.out}; fitted on {len(counts)} frames")
