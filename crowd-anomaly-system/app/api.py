"""FastAPI interface for inference."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from run import run_full_pipeline

app = FastAPI(title="Crowd Anomaly Detection API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
async def infer(video: UploadFile = File(...)):
    tmp_path = Path("tmp_upload.mp4")
    content = await video.read()
    tmp_path.write_bytes(content)
    result = run_full_pipeline(str(tmp_path))
    tmp_path.unlink(missing_ok=True)
    return result
