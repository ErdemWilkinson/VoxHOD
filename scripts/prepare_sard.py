"""SARD (Zenodo 24034563, CC BY 4.0) -> 160x160 tiles in the same .npz format as prepare_heridal.py.

SARD: 1,981 RGB frames (1920x1080) from 35 minutes of drone video, persons labeled
(`SARD/sard_person_labels.csv`: filename, width, height, class, xmin, ymin, xmax, ymax).
SARD has no official split, and consecutive frames are near-duplicates, so frames are split
by number: the first 80% (by frame number) are training frames, the last 20% are a held-out
test block.

Scale is matched to HERIDAL: for every tile the crop size is chosen so that one person's larger
side lands on a size drawn from the HERIDAL training tiles (the reference npz). Per training
frame: 2 tiles centred near a random person and 1 random tile. Test frames are cut into a regular
grid of square crops sized so the median person matches the HERIDAL test median (16.5 px).

    python scripts/prepare_sard.py <SARD folder> <voxhod_g/train.npz> <out folder>

Writes <out>/train.npz and <out>/test.npz.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 160
POSITIVE_PER_FRAME, RANDOM_PER_FRAME = 2, 1
TEST_FRACTION = 0.2
TEST_TARGET = 16.5  # HERIDAL test median of max(w, h) in tile pixels


def read_labels(root: Path) -> dict[str, np.ndarray]:
    boxes = defaultdict(list)
    with open(root / "sard_person_labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            boxes[row["filename"]].append([float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
    return {name: np.array(b, np.float32).reshape(-1, 4) for name, b in boxes.items()}


def cut(image: Image.Image, boxes: np.ndarray, x0: int, y0: int, size: int):
    tile = np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE), Image.BILINEAR))
    if not len(boxes):
        return tile, np.zeros((0, 4), np.float32)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (centres[:, 0] >= x0) & (centres[:, 0] < x0 + size) & (centres[:, 1] >= y0) & (centres[:, 1] < y0 + size)
    kept = np.clip(boxes[inside] - [x0, y0, x0, y0], 0, size) * (TILE / size)
    return tile, kept.astype(np.float32)


def save(out: Path, split: str, tiles: list, boxes: list) -> None:
    counts = np.array([len(b) for b in boxes], np.int32)
    flat = np.concatenate(boxes) if counts.sum() else np.zeros((0, 4), np.float32)
    np.savez(out / f"{split}.npz", images=np.stack(tiles).astype(np.uint8), boxes=flat, counts=counts)
    print(f"{split}: {len(tiles)} tiles, {int(counts.sum())} persons, {int((counts > 0).sum())} tiles with a person", flush=True)


def main() -> None:
    root, ref, out = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rb = ref["boxes"]
    ref_sizes = np.maximum(rb[:, 2] - rb[:, 0], rb[:, 3] - rb[:, 1])
    labels = read_labels(root)
    frames = sorted((p.name for p in root.glob("*.jpg")), key=lambda n: int(n[3:-4]))
    cut_at = int(len(frames) * (1 - TEST_FRACTION))
    rng = np.random.default_rng(0)

    tiles, tile_boxes, skipped = [], [], 0
    for name in frames[:cut_at]:
        image = Image.open(root / name).convert("RGB")
        width, height = image.size
        boxes = labels.get(name, np.zeros((0, 4), np.float32))
        person = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) if len(boxes) else None
        for _ in range(POSITIVE_PER_FRAME if len(boxes) else 0):
            i = int(rng.integers(len(boxes)))
            size = int(round(person[i] * TILE / rng.choice(ref_sizes)))
            if size < TILE // 2 or size > min(width, height):
                skipped += 1
                continue
            cx, cy = boxes[i].reshape(2, 2).mean(0) + rng.uniform(-0.4, 0.4, 2) * size
            x0 = int(np.clip(cx - size / 2, 0, width - size)); y0 = int(np.clip(cy - size / 2, 0, height - size))
            tile, kept = cut(image, boxes, x0, y0, size)
            tiles.append(tile); tile_boxes.append(kept)
        for _ in range(RANDOM_PER_FRAME):
            ref_person = np.median(person) if person is not None else 50.0
            size = int(np.clip(round(ref_person * TILE / rng.choice(ref_sizes)), TILE // 2, min(width, height)))
            x0 = int(rng.integers(0, width - size + 1)); y0 = int(rng.integers(0, height - size + 1))
            tile, kept = cut(image, boxes, x0, y0, size)
            tiles.append(tile); tile_boxes.append(kept)
    print(f"skipped {skipped} positive tiles whose crop would not fit", flush=True)
    save(out, "train", tiles, tile_boxes)

    test_frames = set(frames[cut_at:])
    all_sizes = np.concatenate([np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) for n, b in labels.items() if n in test_frames])
    grid = int(round(np.median(all_sizes) * TILE / TEST_TARGET))
    tiles, tile_boxes = [], []
    for name in frames[cut_at:]:
        image = Image.open(root / name).convert("RGB")
        width, height = image.size
        boxes = labels.get(name, np.zeros((0, 4), np.float32))
        xs = list(range(0, width - grid, grid)) + [width - grid]
        ys = list(range(0, height - grid, grid)) + [height - grid]
        for y0 in ys:
            for x0 in xs:
                tile, kept = cut(image, boxes, x0, y0, grid)
                tiles.append(tile); tile_boxes.append(kept)
    print(f"test grid crop {grid} px over {len(frames) - cut_at} frames", flush=True)
    save(out, "test", tiles, tile_boxes)


if __name__ == "__main__":
    main()
