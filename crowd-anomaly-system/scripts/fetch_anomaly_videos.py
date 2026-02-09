"""Download multiple test videos from URL manifest for benchmarking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


def fetch(manifest: Path, out_dir: Path) -> None:
    items = json.loads(manifest.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        url = item["url"]
        filename = item.get("filename") or Path(url).name
        dst = out_dir / filename
        print(f"Downloading {url} -> {dst}")
        urlretrieve(url, dst)


def write_example_manifest(path: Path) -> None:
    sample = [
        {
            "filename": "example_normal.mp4",
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
        },
        {
            "filename": "example_anomaly_like.mp4",
            "url": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/tracking/faceocc2.webm",
        },
    ]
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Wrote sample manifest: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/video_manifest.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--init_manifest", action="store_true")
    args = parser.parse_args()

    if args.init_manifest:
        write_example_manifest(args.manifest)
    else:
        fetch(args.manifest, args.out_dir)
