"""Video pre-processing helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Tuple

import cv2


def read_video_frames(video_path: str | Path) -> Generator[Tuple[int, any], None, None]:
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def video_metadata(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def extract_frames(video_path: str | Path, out_dir: str | Path, every_n: int = 1) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, frame in read_video_frames(video_path):
        if idx % every_n != 0:
            continue
        out_path = out_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    return saved


def write_video(frames: List, out_path: str | Path, fps: float = 25.0) -> None:
    if not frames:
        raise ValueError("No frames provided")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
