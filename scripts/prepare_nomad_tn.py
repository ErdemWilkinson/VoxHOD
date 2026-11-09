"""NOMAD true-negative frames (CC BY 4.0, github.com/ArtRuss/NOMAD) -> empty 160x160 test tiles.

TN.zip (3.4 GB) holds 500 frames of 5472x3078 with nobody in them, 100 at each flight
distance (10, 30, 50, 70, 90 m). Every detection on these tiles is a false alarm, so they
measure false alarms per tile by altitude on a camera and terrain VoxHOD never saw.

From each frame, <per_frame> random 640 px crops are resized to 160 like the HERIDAL test
tiles. Output: one .npz with images, boxes (empty), counts (zeros) and distance (m), in the
same layout as the other VoxHOD splits.

    python scripts/prepare_nomad_tn.py <TN.zip> <out.npz> [per_frame=8]
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

TILE, CROP = 160, 640


def main() -> None:
    zip_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    per_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rng = np.random.default_rng(11)
    tiles, dist = [], []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
        for i, name in enumerate(names):
            meters = int(Path(name).name.split("_")[1][1:])
            image = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            W, H = image.size
            for _ in range(per_frame):
                x0, y0 = int(rng.integers(0, W - CROP + 1)), int(rng.integers(0, H - CROP + 1))
                tiles.append(np.asarray(image.crop((x0, y0, x0 + CROP, y0 + CROP)).resize((TILE, TILE), Image.BILINEAR)))
                dist.append(meters)
            if i % 100 == 0:
                print(f"  {i}/{len(names)} frames", flush=True)
    np.savez(out, images=np.stack(tiles).astype(np.uint8), boxes=np.zeros((0, 4), np.float32),
             counts=np.zeros(len(tiles), np.int32), distance=np.array(dist, np.int16))
    print(f"{out}: {len(tiles)} empty tiles from {len(names)} frames")


if __name__ == "__main__":
    main()
