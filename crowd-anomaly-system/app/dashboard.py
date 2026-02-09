"""Streamlit dashboard for crowd anomaly visualization."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from run import run_full_pipeline

st.set_page_config(page_title="Crowd Anomaly Dashboard", layout="wide")
st.title("Crowd Anomaly Detection")

uploaded = st.file_uploader("Upload surveillance video", type=["mp4", "avi", "mov"])
if uploaded is not None:
    tmp = Path("dashboard_upload.mp4")
    tmp.write_bytes(uploaded.read())

    with st.spinner("Running full anomaly pipeline..."):
        result = run_full_pipeline(str(tmp))

    st.success("Inference completed")
    st.json(result["summary"])
    st.line_chart(
        {
            "motion": result["scores"]["motion"],
            "trajectory": result["scores"]["trajectory"],
            "density": result["scores"]["density"],
            "final": result["scores"]["final"],
        }
    )
    tmp.unlink(missing_ok=True)
