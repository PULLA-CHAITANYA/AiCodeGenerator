"""Density map inference wrapper around CSRNet."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.density.csrnet import CSRNet


class DensityEstimator:
    def __init__(self, checkpoint: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSRNet(load_weights=False).to(self.device)
        if checkpoint and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
        tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
        pred = self.model(tensor).squeeze().cpu().numpy()
        count = float(pred.sum())
        pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
        return pred, count
