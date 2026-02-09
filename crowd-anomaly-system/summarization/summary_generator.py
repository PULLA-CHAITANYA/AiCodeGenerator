"""Generate machine- and human-readable anomaly summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from summarization.event_segmenter import EventSegment
from summarization.pattern_classifier import PatternResult


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class SummaryGenerator:
    def build_timeline(
        self,
        segments: List[EventSegment],
        segment_patterns: List[PatternResult],
        module_scores: Dict[str, List[float]],
    ) -> List[dict]:
        timeline = []
        for seg, pat in zip(segments, segment_patterns):
            motion_mean = float(np.mean(module_scores["motion"][seg.start_frame : seg.end_frame + 1]))
            traj_mean = float(np.mean(module_scores["trajectory"][seg.start_frame : seg.end_frame + 1]))
            density_mean = float(np.mean(module_scores["density"][seg.start_frame : seg.end_frame + 1]))
            timeline.append(
                {
                    "start_time": _format_time(seg.start_time),
                    "end_time": _format_time(seg.end_time),
                    "anomaly_type": pat.anomaly_type,
                    "pattern_label": pat.pattern,
                    "severity": seg.severity,
                    "contributing_modules": pat.contributors,
                    "confidence_score": round(float(pat.confidence), 3),
                    "module_means": {
                        "motion": round(motion_mean, 3),
                        "trajectory": round(traj_mean, 3),
                        "density": round(density_mean, 3),
                    },
                }
            )
        return timeline

    @staticmethod
    def save_timeline_json(timeline: List[dict], out_path: str | Path) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

    @staticmethod
    def save_summary_text(timeline: List[dict], density_counts: List[float], out_path: str | Path) -> None:
        lines = []
        baseline = float(np.mean(density_counts[: max(1, len(density_counts) // 5)])) if density_counts else 0.0
        for item in timeline:
            density_mean = item["module_means"]["density"]
            spike_pct = ((density_mean - baseline) / (abs(baseline) + 1e-6)) * 100 if baseline > 0 else 0.0
            lines.append(
                f"Between {item['start_time']}–{item['end_time']}, {item['pattern_label'].lower()} was detected. "
                f"Severity was {item['severity']} with confidence {item['confidence_score']:.2f}. "
                f"Module contributions: {', '.join(item['contributing_modules'])}. "
                f"Estimated density deviation: {spike_pct:.1f}% from baseline."
            )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) if lines else "No notable events detected.")

    @staticmethod
    def save_graphs(module_scores: Dict[str, List[float]], timeline: List[dict], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 5))
        for name, vals in module_scores.items():
            plt.plot(vals, label=name)
        plt.title("Module Score Trends")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "module_scores.png")
        plt.close()

        severity_map = {"normal": 0, "warning": 1, "critical": 2}
        sev_series = []
        for item in timeline:
            sev = severity_map.get(item["severity"], 0)
            start = int(item["start_time"].split(":")[0]) * 60 + int(item["start_time"].split(":")[1])
            end = int(item["end_time"].split(":")[0]) * 60 + int(item["end_time"].split(":")[1])
            sev_series.extend([sev] * max(1, end - start + 1))

        if sev_series:
            plt.figure(figsize=(12, 3))
            plt.plot(sev_series)
            plt.yticks([0, 1, 2], ["normal", "warning", "critical"])
            plt.title("Severity Timeline (second-level)")
            plt.tight_layout()
            plt.savefig(out_dir / "severity_timeline.png")
            plt.close()
