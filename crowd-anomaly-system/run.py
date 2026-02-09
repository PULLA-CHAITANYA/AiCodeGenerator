"""Main entrypoint for full crowd anomaly inference and summarization."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pipelines.density_pipeline import DensityPipeline
from pipelines.ensemble import EnsembleAnomalyScorer
from pipelines.motion_pipeline import MotionPipeline
from pipelines.trajectory_pipeline import TrajectoryPipeline
from summarization.event_segmenter import EventSegmenter
from summarization.pattern_classifier import PatternClassifier
from summarization.summary_generator import SummaryGenerator
from summarization.video_annotator import VideoAnnotator
from utils.config import CONFIG, ensure_directories
from utils.video_utils import write_video
from utils.visualization import plot_scores, write_alert_log


def _safe_trajectory(video_path: str) -> dict:
    try:
        traj_pipe = TrajectoryPipeline(model_path=str(CONFIG.paths.checkpoints_dir / "trajectory_iforest.joblib"))
        return traj_pipe.run_video(video_path)
    except Exception as exc:  # fallback for environments missing YOLO/DeepSORT
        print(f"Trajectory pipeline warning: {exc}")
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            "scores": [0.0] * max(0, n - 1),
            "tracks": [[] for _ in range(max(0, n - 1))],
            "feature_stats": [{"mean_speed": 0.0, "mean_acceleration": 0.0, "mean_direction_change": 0.0, "mean_curvature": 0.0} for _ in range(max(0, n - 1))],
        }


def _safe_density(video_path: str) -> dict:
    try:
        density_pipe = DensityPipeline(detector_path=str(CONFIG.paths.checkpoints_dir / "density_iforest.joblib"))
        return density_pipe.run_video(video_path)
    except Exception as exc:
        print(f"Density pipeline warning: {exc}")
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {"counts": [0.0] * max(0, n - 1), "scores": [0.0] * max(0, n - 1)}


def run_full_pipeline(video_path: str) -> dict:
    ensure_directories()

    motion_pipe = MotionPipeline()
    motion_scores, rendered_motion = motion_pipe.run_video(video_path)

    traj_out = _safe_trajectory(video_path)
    density_out = _safe_density(video_path)

    fusion = EnsembleAnomalyScorer()
    fused = fusion.fuse(motion_scores, traj_out["scores"], density_out["scores"])

    cap_meta = cv2.VideoCapture(video_path)
    fps = cap_meta.get(cv2.CAP_PROP_FPS) or 25.0
    cap_meta.release()
    segmenter = EventSegmenter()
    segments = segmenter.segment(fused["final"], fps)

    classifier = PatternClassifier()
    segment_patterns = []
    for seg in segments:
        start, end = seg.start_frame, seg.end_frame + 1
        segment_patterns.append(
            classifier.classify(
                motion_scores=fused["motion_norm"][start:end],
                trajectory_scores=fused["trajectory_norm"][start:end],
                density_scores=fused["density_norm"][start:end],
                density_counts=density_out["counts"][start:end] if density_out.get("counts") else [],
                trajectory_features=traj_out.get("feature_stats", [])[start:end],
            )
        )

    timeline_generator = SummaryGenerator()
    scores = {
        "motion": fused["motion_norm"],
        "trajectory": fused["trajectory_norm"],
        "density": fused["density_norm"],
        "final": fused["final"],
    }
    timeline = timeline_generator.build_timeline(segments, segment_patterns, scores)

    severity_per_frame = ["normal"] * len(fused["final"])
    pattern_per_frame = ["Normal flow"] * len(fused["final"])
    for seg, pat in zip(segments, segment_patterns):
        for idx in range(seg.start_frame, seg.end_frame + 1):
            if idx < len(severity_per_frame):
                severity_per_frame[idx] = seg.severity
                pattern_per_frame[idx] = pat.pattern

    n_alerts = int(sum(fused["anomaly"]))
    summary = {
        "frames": len(fused["final"]),
        "anomaly_frames": n_alerts,
        "anomaly_ratio": float(n_alerts / max(1, len(fused["final"]))),
        "events": len(segments),
    }

    return {
        "summary": summary,
        "scores": scores,
        "flags": fused["anomaly"],
        "rendered": rendered_motion,
        "timeline": timeline,
        "severity_per_frame": severity_per_frame,
        "pattern_per_frame": pattern_per_frame,
        "trajectory_tracks": traj_out.get("tracks", []),
        "density_scores_raw": density_out.get("scores", []),
        "density_counts": density_out.get("counts", []),
    }


def main(video_path: str) -> None:
    result = run_full_pipeline(video_path)

    out_dir = CONFIG.paths.outputs_dir
    graphs_dir = out_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if result["rendered"]:
        write_video(result["rendered"], out_dir / "motion_overlay.mp4")

    plot_scores(result["scores"], out_dir / "score_plot.png")

    alerts = [
        {"frame": i, "score": s, "anomaly": f}
        for i, (s, f) in enumerate(zip(result["scores"]["final"], result["flags"]))
        if f == 1
    ]
    write_alert_log(alerts, out_dir / "alerts.log")

    timeline_generator = SummaryGenerator()
    timeline_generator.save_timeline_json(result["timeline"], out_dir / "anomaly_timeline.json")
    timeline_generator.save_summary_text(result["timeline"], result["density_counts"], out_dir / "summary.txt")
    timeline_generator.save_graphs(result["scores"], result["timeline"], graphs_dir)

    annotator = VideoAnnotator()
    annotator.annotate(
        video_path=video_path,
        output_path=out_dir / "annotated_video.mp4",
        fused_scores=result["scores"]["final"],
        severity_labels=result["severity_per_frame"],
        pattern_labels=result["pattern_per_frame"],
        trajectory_tracks=result["trajectory_tracks"],
        density_scores=result["density_scores_raw"],
    )

    print("Summary:", result["summary"])
    print(f"Artifacts saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    args = parser.parse_args()
    main(args.video)
