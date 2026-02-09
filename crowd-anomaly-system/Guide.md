# Guide: Run, Connect, and Test the Crowd Anomaly Detection System

This guide explains how to run the full system locally, how the modules connect, and how to validate outputs.

## 1) System Architecture (How everything connects)

The runtime flow is:

1. **Input video** (`run.py --video ...`)
2. **Motion pipeline**
   - Optical flow from adjacent frames
   - GAN reconstruction error => motion anomaly score
3. **Trajectory pipeline**
   - YOLOv8 detects people
   - DeepSORT tracks IDs
   - Trajectory features (speed, acceleration, direction change, curvature)
   - Isolation Forest => trajectory anomaly score
4. **Density pipeline**
   - CSRNet estimates density map/count per frame
   - Isolation Forest => density anomaly score
5. **Ensemble fusion**
   - Normalize module scores
   - Weighted fusion => final anomaly score + flag
6. **Summarization module**
   - Segment timeline into normal/warning/critical events
   - Deterministic pattern classification
   - Produce JSON timeline + human summary + graphs
7. **Annotation output**
   - Annotated video overlays (flow heatmap, trajectories, labels)

All of this is orchestrated by `run.py`.

---

## 2) Prerequisites

- Python 3.10+
- Optional CUDA GPU (CPU also works)
- A small set of test videos (normal + test)

---

## 3) Setup

```bash
cd crowd-anomaly-system
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4) Prepare folder inputs

Place videos under:

```text
data/raw_videos/
  normal_video.mp4
  test_video.mp4
```

You can include multiple videos; preparation scans `.mp4` and `.avi`.

---

## 5) Data preparation (frames + optical flow)

```bash
python scripts/prepare_dataset.py --raw_video_dir data/raw_videos
```

Expected outputs:
- `data/frames/<video_name>/frame_*.jpg`
- `data/optical_flow/<video_name>.npy`

---

## 6) Train required models (normal behavior baseline)

### 6.1 Motion model (GAN)
```bash
python -m models.optical_flow_gan.train --normal_flow_dir data/optical_flow --epochs 20
```
Produces: `checkpoints/flow_gan.pt`

### 6.2 Trajectory anomaly model
```bash
python scripts/train_trajectory.py --video data/raw_videos/normal_video.mp4 --out checkpoints/trajectory_iforest.joblib
```

### 6.3 Density anomaly model
```bash
python scripts/train_density.py --video data/raw_videos/normal_video.mp4 --out checkpoints/density_iforest.joblib
```

---

## 7) Run full inference + summarization

```bash
python run.py --video data/raw_videos/test_video.mp4
```

This command runs all 3 detection modules, fuses scores, segments events, classifies patterns, and writes output artifacts.

---

## 8) Validate outputs (what to check)

After inference, verify:

```bash
ls -lah outputs
ls -lah outputs/graphs
```

Required outputs:
- `outputs/annotated_video.mp4`
- `outputs/anomaly_timeline.json`
- `outputs/summary.txt`
- `outputs/graphs/module_scores.png`
- `outputs/graphs/severity_timeline.png`

Additional outputs:
- `outputs/motion_overlay.mp4`
- `outputs/score_plot.png`
- `outputs/alerts.log`

Inspect timeline JSON:

```bash
python -m json.tool outputs/anomaly_timeline.json | head -n 80
```

---

## 9) How to test module-by-module

### Motion-only sanity
- Ensure `checkpoints/flow_gan.pt` exists.
- Run full pipeline and inspect `motion_overlay.mp4` for visible high-motion hotspot regions.

### Trajectory-only sanity
- Ensure environment has `ultralytics` + `deep-sort-realtime`.
- Ensure `checkpoints/trajectory_iforest.joblib` exists.
- Run full pipeline and inspect `annotated_video.mp4` for stable track IDs/paths.

### Density-only sanity
- Ensure `checkpoints/density_iforest.joblib` exists.
- Run full pipeline and inspect density-related spikes in `anomaly_timeline.json` and `summary.txt`.

---

## 10) API and Dashboard integration

### FastAPI
Start API:
```bash
uvicorn app.api:app --reload --port 8000
```
Health check:
```bash
curl -s http://127.0.0.1:8000/health
```
Inference request (multipart upload):
```bash
curl -X POST "http://127.0.0.1:8000/infer" -F "video=@data/raw_videos/test_video.mp4"
```

### Streamlit dashboard
```bash
streamlit run app/dashboard.py
```
Then upload a test video and inspect returned summary + score chart.

---

## 11) Common issues and fixes

1. **`ModuleNotFoundError` (numpy/torch/etc.)**
   - Activate venv and reinstall requirements:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **YOLO / DeepSORT not working**
   - Ensure the packages are installed and supported on your OS/Python version.
   - System has fallbacks for missing optional trajectory dependencies, but richer annotations require these modules.

3. **No anomalies detected**
   - Re-check training data: use truly normal footage for baseline.
   - Tune thresholds/weights in `utils/config.py`.

4. **Performance is slow**
   - Use shorter test clips first.
   - Use GPU-enabled torch build.

---

## 12) Minimal quick-start (copy/paste)

```bash
cd crowd-anomaly-system
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python scripts/prepare_dataset.py --raw_video_dir data/raw_videos
python -m models.optical_flow_gan.train --normal_flow_dir data/optical_flow --epochs 5
python scripts/train_trajectory.py --video data/raw_videos/normal_video.mp4 --out checkpoints/trajectory_iforest.joblib
python scripts/train_density.py --video data/raw_videos/normal_video.mp4 --out checkpoints/density_iforest.joblib
python run.py --video data/raw_videos/test_video.mp4
```

Then open:
- `outputs/annotated_video.mp4`
- `outputs/anomaly_timeline.json`
- `outputs/summary.txt`
