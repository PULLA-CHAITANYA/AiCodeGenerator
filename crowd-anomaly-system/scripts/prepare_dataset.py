"""Prepare frames and optical flow files from raw videos."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import CONFIG, ensure_directories
from utils.flow_utils import generate_flow_stack, save_flow_npy
from utils.video_utils import extract_frames


def prepare(raw_video_dir: Path) -> None:
    ensure_directories()
    videos = sorted(raw_video_dir.glob("*.mp4")) + sorted(raw_video_dir.glob("*.avi"))
    if not videos:
        print("No raw videos found.")
        return

    for video in videos:
        stem = video.stem
        frame_dir = CONFIG.paths.frames_dir / stem
        flow_out = CONFIG.paths.optical_flow_dir / f"{stem}.npy"

        extract_frames(video, frame_dir, every_n=1)
        flows = generate_flow_stack(video, resize=(CONFIG.motion.image_size, CONFIG.motion.image_size))
        save_flow_npy(flows, flow_out)
        print(f"Prepared: {video.name} -> {frame_dir} and {flow_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_video_dir", type=Path, default=CONFIG.paths.raw_videos_dir)
    args = parser.parse_args()
    prepare(args.raw_video_dir)
