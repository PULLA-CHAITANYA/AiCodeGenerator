"""Main entrypoint for full crowd anomaly inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.density_pipeline import DensityPipeline
from pipelines.ensemble import EnsembleAnomalyScorer
from pipelines.motion_pipeline import MotionPipeline
from pipelines.trajectory_pipeline import TrajectoryPipeline
from utils.config import CONFIG, ensure_directories
from utils.video_utils import write_video
from utils.visualization import plot_scores, write_alert_log


def run_full_pipeline(video_path: str) -> dict:
    ensure_directories()

    motion_pipe = MotionPipeline()
    motion_scores, rendered_motion = motion_pipe.run_video(video_path)

    traj_pipe = TrajectoryPipeline(model_path=str(CONFIG.paths.checkpoints_dir / "trajectory_iforest.joblib"))
    traj_out = traj_pipe.run_video(video_path)

    density_pipe = DensityPipeline(detector_path=str(CONFIG.paths.checkpoints_dir / "density_iforest.joblib"))
    density_out = density_pipe.run_video(video_path)

    fusion = EnsembleAnomalyScorer()
    fused = fusion.fuse(motion_scores, traj_out["scores"], density_out["scores"])

    n_alerts = int(sum(fused["anomaly"]))
    summary = {
        "frames": len(fused["final"]),
        "anomaly_frames": n_alerts,
        "anomaly_ratio": float(n_alerts / max(1, len(fused["final"]))),
    }

    scores = {
        "motion": fused["motion_norm"],
        "trajectory": fused["trajectory_norm"],
        "density": fused["density_norm"],
        "final": fused["final"],
    }

    return {"summary": summary, "scores": scores, "flags": fused["anomaly"], "rendered": rendered_motion}


def main(video_path: str) -> None:
    result = run_full_pipeline(video_path)

    out_dir = CONFIG.paths.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if result["rendered"]:
        write_video(result["rendered"], out_dir / "motion_overlay.mp4")

    plot_scores(result["scores"], out_dir / "score_plot.png")

    alerts = [
        {"frame": i, "score": s, "anomaly": f}
        for i, (s, f) in enumerate(zip(result["scores"]["final"], result["flags"]))
        if f == 1
    ]
    write_alert_log(alerts, out_dir / "alerts.log")

    print("Summary:", result["summary"])
    print(f"Artifacts saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    args = parser.parse_args()
    main(args.video)
