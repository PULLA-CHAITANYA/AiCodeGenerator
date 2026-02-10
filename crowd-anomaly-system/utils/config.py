"""Centralized configuration for crowd anomaly system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_videos_dir: Path = data_dir / "raw_videos"
    frames_dir: Path = data_dir / "frames"
    optical_flow_dir: Path = data_dir / "optical_flow"
    density_maps_dir: Path = data_dir / "density_maps"
    checkpoints_dir: Path = project_root / "checkpoints"
    outputs_dir: Path = project_root / "outputs"


@dataclass
class MotionConfig:
    image_size: int = 128
    sequence_length: int = 4
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 20
    lambda_l1: float = 50.0


@dataclass
class TrajectoryConfig:
    yolo_model: str = "yolov8n.pt"
    deepsort_max_age: int = 30
    track_history: int = 64
    anomaly_contamination: float = 0.05


@dataclass
class DensityConfig:
    input_size: int = 256
    isolation_contamination: float = 0.07
    overcrowding_threshold: float = 120.0


@dataclass
class EnsembleConfig:
    weights: Dict[str, float] = field(
        default_factory=lambda: {"motion": 0.4, "trajectory": 0.35, "density": 0.25}
    )
    anomaly_threshold: float = 0.65


@dataclass
class SystemConfig:
    paths: Paths = field(default_factory=Paths)
    motion: MotionConfig = field(default_factory=MotionConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


CONFIG = SystemConfig()


def ensure_directories() -> None:
    """Create runtime directories used by the project."""
    for directory in [
        CONFIG.paths.raw_videos_dir,
        CONFIG.paths.frames_dir,
        CONFIG.paths.optical_flow_dir,
        CONFIG.paths.density_maps_dir,
        CONFIG.paths.checkpoints_dir,
        CONFIG.paths.outputs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
