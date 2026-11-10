"""Split the NOMAD empty tiles into a training half and a test half.

nomad_tn.npz (from prepare_nomad_tn.py) holds 8 tiles per frame, frames in file-name
order, 100 frames at each flight distance (10/30/50/70/90 m). Within each distance the
first 50 frames go to training and the last 50 to test, so neighbouring frames of the
same flight do not end up on both sides of the split as often as a random split would.
The training half (2,000 empty tiles) is appended to <base_dir>/train.npz.

    python scripts/prepare_nomad_split.py <nomad_tn.npz> <base_dir> <out_dir>

<out_dir> gets train.npz (base + NOMAD training half), nomad_tn_test.npz (test half, with
distance) and copies of val.npz / test.npz / wisard.npz from <base_dir> if present.
<base_dir> also gets nomad_tn_test.npz, so models trained without NOMAD are scored on the
same tiles.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PER_FRAME = 8


def main() -> None:
    source, base, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    nomad = np.load(source)
    distance = nomad["distance"]
    frames = len(distance) // PER_FRAME
    frame_distance = distance[::PER_FRAME]
    rank = np.zeros(frames, int)
    seen: dict[int, int] = {}
    for f, d in enumerate(frame_distance):
        rank[f] = seen.get(int(d), 0)
        seen[int(d)] = rank[f] + 1
    half = {d: n // 2 for d, n in seen.items()}
    train_frame = np.array([rank[f] < half[int(frame_distance[f])] for f in range(frames)])
    train_tile = np.repeat(train_frame, PER_FRAME)

    test_images = nomad["images"][~train_tile]
    for folder in (out, base):
        np.savez(folder / "nomad_tn_test.npz", images=test_images, boxes=np.zeros((0, 4), np.float32),
                 counts=np.zeros(len(test_images), np.int32), distance=distance[~train_tile])
    print(f"NOMAD: {int(train_tile.sum())} training tiles, {len(test_images)} test tiles")

    data = np.load(base / "train.npz")
    added = nomad["images"][train_tile]
    np.savez(out / "train.npz", images=np.concatenate([data["images"], added]), boxes=data["boxes"],
             counts=np.concatenate([data["counts"], np.zeros(len(added), np.int32)]))
    print(f"train: {len(data['images'])} + {len(added)} NOMAD tiles")
    for name in ("val.npz", "test.npz", "wisard.npz"):
        if (base / name).exists():
            shutil.copyfile(base / name, out / name)


if __name__ == "__main__":
    main()
