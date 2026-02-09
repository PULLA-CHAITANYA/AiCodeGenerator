"""Inference utilities for motion-based anomaly scores."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.optical_flow_gan.generator import FlowGenerator
from utils.config import CONFIG
from utils.flow_utils import compute_farneback_flow, flow_to_mag_angle


class MotionAnomalyDetector:
    def __init__(self, checkpoint: Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlowGenerator().to(self.device)
        ckpt = checkpoint or (CONFIG.paths.checkpoints_dir / "flow_gan.pt")
        if ckpt.exists():
            data = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(data["generator"])
        self.model.eval()

    @torch.no_grad()
    def score_flow(self, flow_two_channel: np.ndarray) -> tuple[float, np.ndarray]:
        x = torch.tensor(flow_two_channel, dtype=torch.float32).unsqueeze(0).to(self.device)
        x = (x - x.mean()) / (x.std() + 1e-6)
        recon = self.model(x).cpu().numpy()[0]
        err_map = np.mean((flow_two_channel - recon) ** 2, axis=0)
        score = float(err_map.mean())
        return score, err_map

    def score_frame_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> tuple[float, np.ndarray]:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = compute_farneback_flow(prev_gray, curr_gray)
        mag, ang = flow_to_mag_angle(flow)
        stacked = np.stack([mag, ang], axis=0).astype(np.float32)
        return self.score_flow(stacked)
