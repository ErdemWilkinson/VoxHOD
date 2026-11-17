"""Empty and hard-negative tiles from unlabeled drone photos (e.g. the Serbia land-cover set, CC BY 4.0).

Step 1 (mine): per photo, CANDIDATES random square crops (sizes from VOX_CROPS, default 160,240,320 px,
which match HERIDAL's tile scale for a ground sampling distance about twice HERIDAL's), resized to 160,
scored by a trained model; keeps the HARD highest-scoring and RAND random ones. Writes <out>/cand.npz and
contact sheets <out>/sheet_NN.png (48 numbered tiles each) for inspection.

    python scripts/mine_external.py mine <photo_dir> <model.keras> <out_dir> [hard] [random]

The photos are not labeled, so the highest-scoring crops are the most likely to contain a real person.
Inspect every sheet, list the numbers to drop (one per line), then:

    python scripts/mine_external.py finish <out_dir> <drop.txt>

which writes <out>/neg.npz (empty tiles, for train.py VOX_EXTRA_TRAIN).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE, CANDIDATES, PER_SHEET, COLS = 160, 40, 48, 8


def sheets(images: np.ndarray, out: Path) -> None:
    for s in range(0, len(images), PER_SHEET):
        chunk = images[s:s + PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (TILE + 4), rows * (TILE + 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, tile in enumerate(chunk):
            x, y = (k % COLS) * (TILE + 4), (k // COLS) * (TILE + 4)
            sheet.paste(Image.fromarray(tile), (x, y))
            draw.rectangle((x, y, x + 34, y + 14), fill="black")
            draw.text((x + 2, y + 1), str(s + k), fill="yellow")
        sheet.save(out / f"sheet_{s // PER_SHEET:02d}.png")


def mine(photo_dir: Path, model_path: str, out: Path, hard_n: int, rand_n: int) -> None:
    import tensorflow as tf

    crops = [int(c) for c in os.environ.get("VOX_CROPS", "160,240,320").split(",")]
    model = tf.keras.models.load_model(model_path, compile=False)
    rng = np.random.default_rng(1)
    photos = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    tiles, meta = [], []
    for index, path in enumerate(photos):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        cands, boxes = [], []
        for _ in range(CANDIDATES):
            size = int(rng.choice(crops))
            x0, y0 = int(rng.integers(0, width - size)), int(rng.integers(0, height - size))
            cands.append(np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE), Image.BILINEAR)))
            boxes.append((x0, y0, size))
        batch = np.stack(cands)
        heat = model.predict(batch.astype(np.float32) / 127.5 - 1.0, verbose=0)[..., 0]
        score = heat.reshape(len(batch), -1).max(axis=1)
        order = np.argsort(-score)
        rand = rng.choice(order[hard_n:], size=min(rand_n, len(order) - hard_n), replace=False)
        for kind, picks in (("hard", order[:hard_n]), ("random", rand)):
            for i in picks:
                tiles.append(batch[i])
                meta.append((path.name, *boxes[i], float(score[i]), kind))
        if index % 25 == 0:
            print(f"  {index}/{len(photos)} photos, {len(tiles)} tiles", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    images = np.stack(tiles).astype(np.uint8)
    np.savez(out / "cand.npz", images=images, meta=np.array(meta, dtype=object))
    sheets(images, out)
    scores = np.array([m[4] for m in meta if m[5] == "hard"])
    print(f"{len(images)} candidate tiles from {len(photos)} photos; hard score median {np.median(scores):.3f}, "
          f"max {scores.max():.3f}; {(len(images) + PER_SHEET - 1) // PER_SHEET} sheets to inspect")


def finish(out: Path, drop_file: str) -> None:
    data = np.load(out / "cand.npz", allow_pickle=True)
    drop = {int(x) for x in Path(drop_file).read_text().split()}
    keep = np.array([i for i in range(len(data["images"])) if i not in drop], dtype=int)
    images = data["images"][keep]
    np.savez(out / "neg.npz", images=images, boxes=np.zeros((0, 4), np.float32), counts=np.zeros(len(images), np.int32))
    print(f"neg: {len(images)} empty tiles kept, {len(drop)} dropped after inspection")


if __name__ == "__main__":
    if sys.argv[1] == "mine":
        mine(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
             int(sys.argv[5]) if len(sys.argv) > 5 else 3, int(sys.argv[6]) if len(sys.argv) > 6 else 1)
    else:
        finish(Path(sys.argv[2]), sys.argv[3])
