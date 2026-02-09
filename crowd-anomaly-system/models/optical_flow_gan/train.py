"""Training script for optical-flow GAN on normal crowd videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.optical_flow_gan.discriminator import FlowDiscriminator
from models.optical_flow_gan.generator import FlowGenerator
from utils.config import CONFIG, ensure_directories


class FlowDataset(Dataset):
    def __init__(self, npy_files: list[Path]):
        self.items = []
        for f in npy_files:
            arr = np.load(f)
            self.items.extend(arr)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x = self.items[idx]
        x = torch.tensor(x, dtype=torch.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, x


def train(normal_flow_dir: Path, epochs: int) -> None:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npy_files = sorted(normal_flow_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError("No .npy optical flow files found")

    ds = FlowDataset(npy_files)
    dl = DataLoader(ds, batch_size=CONFIG.motion.batch_size, shuffle=True, num_workers=0)

    g = FlowGenerator().to(device)
    d = FlowDiscriminator().to(device)

    adv = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    g_opt = torch.optim.Adam(g.parameters(), lr=CONFIG.motion.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=CONFIG.motion.lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            valid = torch.ones((x.size(0), 1, 15, 15), device=device)
            fake = torch.zeros((x.size(0), 1, 15, 15), device=device)

            with torch.no_grad():
                y_fake_detached = g(x)
            d_real = d(x, y)
            d_fake = d(x, y_fake_detached)
            d_loss = 0.5 * (adv(d_real, valid) + adv(d_fake, fake))
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            y_fake = g(x)
            g_adv = adv(d(x, y_fake), valid)
            g_recon = l1(y_fake, y)
            g_loss = g_adv + CONFIG.motion.lambda_l1 * g_recon
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        print(f"Epoch {epoch}/{epochs} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

    ckpt = CONFIG.paths.checkpoints_dir / "flow_gan.pt"
    torch.save({"generator": g.state_dict(), "discriminator": d.state_dict()}, ckpt)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_flow_dir", type=Path, default=CONFIG.paths.optical_flow_dir)
    parser.add_argument("--epochs", type=int, default=CONFIG.motion.epochs)
    args = parser.parse_args()
    train(args.normal_flow_dir, args.epochs)
