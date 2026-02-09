# Crowd Anomaly Detection System

Production-style, modular, end-to-end pipeline for abnormal crowd behavior detection using three complementary methods:

1. **Motion-based (Optical Flow + GAN reconstruction)**
2. **Trajectory-based (YOLOv8 + DeepSORT + Isolation Forest)**
3. **Density-based (CSRNet + Isolation Forest time-series analysis)**

---

## 1) Project Structure

```text
crowd-anomaly-system/
│── data/
│   ├── raw_videos/
│   ├── frames/
│   ├── optical_flow/
│   └── density_maps/
│
│── models/
│   ├── optical_flow_gan/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── trajectory/
│   │   ├── yolo_detector.py
│   │   ├── deepsort_tracker.py
│   │   ├── trajectory_features.py
│   │   └── anomaly_detector.py
│   └── density/
│       ├── csrnet.py
│       ├── density_estimator.py
│       └── density_anomaly.py
│
│── pipelines/
│   ├── motion_pipeline.py
│   ├── trajectory_pipeline.py
│   ├── density_pipeline.py
│   └── ensemble.py
│
│── utils/
│   ├── video_utils.py
│   ├── flow_utils.py
│   ├── visualization.py
│   └── config.py
│
│── scripts/
│   ├── prepare_dataset.py
│   ├── train_trajectory.py
│   └── train_density.py
│
│── app/
│   ├── api.py
│   └── dashboard.py
│
│── requirements.txt
│── run.py
│── README.md
```

---

## 2) Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA optional (inference/training works on CPU)

### Install

```bash
cd crowd-anomaly-system
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Dataset Suggestions

Use any surveillance-like crowd datasets:
- UCSD Pedestrian (anomaly baseline)
- UMN Crowd dataset (panic/running events)
- ShanghaiTech Campus (crowd counts)
- UCF-Crime (for broader abnormal events)

### Recommended data split
- `data/raw_videos/normal/*.mp4` for model fitting on normal behavior
- `data/raw_videos/test/*.mp4` for anomaly inference

---

## 4) Data Preparation

Extract frames and pre-compute optical flow stacks:

```bash
python scripts/prepare_dataset.py --raw_video_dir data/raw_videos
```

Outputs:
- JPEG frames under `data/frames/<video_name>/`
- Flow tensors (`.npy`) under `data/optical_flow/`

---

## 5) Train Each Module

### 5.1 Motion GAN (normal motion only)

```bash
python -m models.optical_flow_gan.train --normal_flow_dir data/optical_flow --epochs 20
```

Output: `checkpoints/flow_gan.pt`

### 5.2 Trajectory anomaly detector (normal trajectories only)

```bash
python scripts/train_trajectory.py --video data/raw_videos/normal_video.mp4 --out checkpoints/trajectory_iforest.joblib
```

### 5.3 Density anomaly detector (normal density baseline)

```bash
python scripts/train_density.py --video data/raw_videos/normal_video.mp4 --out checkpoints/density_iforest.joblib
```

---

## 6) Run Inference

Run complete ensemble pipeline on a target video:

```bash
python run.py --video data/raw_videos/test_video.mp4
```

Generated artifacts in `outputs/`:
- `motion_overlay.mp4` (frame-level motion heatmaps)
- `score_plot.png` (module + fused scores)
- `alerts.log` (anomaly frame events)

---

## 7) Visualization & Alerts

- Motion heatmap overlay highlights high flow-reconstruction error regions.
- Score plot compares motion/trajectory/density/final fused score.
- Final anomaly decision uses weighted fusion from `pipelines/ensemble.py`.

---

## 8) Optional REST API

```bash
uvicorn app.api:app --reload --port 8000
```

- `GET /health`
- `POST /infer` with uploaded video file

---

## 9) Optional Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Upload a video and inspect time-series anomaly scores interactively.

---

## 10) Notes for Production

- Swap Farneback with RAFT for higher motion quality.
- Add calibration per camera view for thresholds.
- Persist metadata and event clips to a DB/message queue.
- Add model versioning for checkpoints and rollback support.
