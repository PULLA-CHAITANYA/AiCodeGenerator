"""Optical flow utilities for motion anomaly detection."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .video_utils import read_video_frames


def compute_farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_to_mag_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    mag, ang = flow_to_mag_angle(flow)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def generate_flow_stack(video_path: str | Path, resize: Tuple[int, int] = (128, 128)) -> List[np.ndarray]:
    flows = []
    prev_gray = None
    for _, frame in read_video_frames(video_path):
        frame = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = compute_farneback_flow(prev_gray, gray)
            mag, ang = flow_to_mag_angle(flow)
            stacked = np.stack([mag, ang], axis=0).astype(np.float32)
            flows.append(stacked)
        prev_gray = gray
    return flows


def save_flow_npy(flows: List[np.ndarray], out_path: str | Path) -> None:
    np.save(str(out_path), np.array(flows, dtype=np.float32))
