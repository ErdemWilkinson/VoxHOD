"""Add empty and hard-negative tiles to a VoxHOD training set.

The 2026-11-07 analysis found 89% of false alarms on empty tiles, while only 29% of
training tiles are empty (91% of test tiles are). This script samples tiles with no
person in them from every HERIDAL training photo, scores them with a trained model,
and adds the highest-scoring ones (hard negatives: places the model wrongly thinks a
person is) plus a few random empty ones. Validation, test and WiSARD stay untouched.

    python scripts/mine_negatives.py <heridal_voc> <base_dir> <model.h5> <out_dir> [hard] [random]

<base_dir> holds the existing train/val/test/wisard .npz; <out_dir> gets the new
train.npz and links to the other splits. Images stay below TensorFlow's 2 GB tensor
limit, which caps how many tiles can be added.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_heridal import CROP_SIZES, TILE, load, split_names  # noqa: E402

CANDIDATES = 40  # empty candidate tiles scored per photo


def main() -> None:
    root, base, model_path, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    hard_n = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    rand_n = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    rng = np.random.default_rng(1)
    new_tiles, scores_kept = [], []
    names = split_names(root, "train")
    for index, name in enumerate(names):
        image, boxes = load(root, name)
        width, height = image.size
        cands = []
        for _ in range(CANDIDATES * 3):
            if len(cands) == CANDIDATES:
                break
            size = int(rng.choice(CROP_SIZES))
            x0, y0 = int(rng.integers(0, width - size)), int(rng.integers(0, height - size))
            # Empty means no person box touches the crop at all, not just no centre inside it.
            if len(boxes) and np.any((boxes[:, 0] < x0 + size) & (boxes[:, 2] > x0) & (boxes[:, 1] < y0 + size) & (boxes[:, 3] > y0)):
                continue
            cands.append(np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE))))
        if not cands:
            continue
        batch = np.stack(cands)
        heat = model.predict(batch.astype(np.float32) / 127.5 - 1.0, verbose=0)[..., 0]
        score = heat.reshape(len(batch), -1).max(axis=1)
        order = np.argsort(-score)
        hard = order[:hard_n]
        rest = order[hard_n:]
        rand = rng.choice(rest, size=min(rand_n, len(rest)), replace=False) if len(rest) else []
        for i in list(hard) + list(rand):
            new_tiles.append(batch[i])
        scores_kept.extend(score[hard].tolist())
        if index % 200 == 0:
            print(f"  {index}/{len(names)} photos, {len(new_tiles)} negatives, median hard score {np.median(scores_kept):.3f}", flush=True)

    data = np.load(base / "train.npz")
    images = np.concatenate([data["images"], np.stack(new_tiles).astype(np.uint8)])
    counts = np.concatenate([data["counts"], np.zeros(len(new_tiles), np.int32)])
    if images.nbytes >= 2**31:
        raise SystemExit(f"{images.nbytes / 2**30:.2f} GiB of images is over the 2 GB tensor limit; lower the counts")
    np.savez(out / "train.npz", images=images, boxes=data["boxes"], counts=counts)
    for split in ("val", "test", "wisard"):
        link = out / f"{split}.npz"
        if not link.exists() and (base / f"{split}.npz").exists():
            os.symlink(base / f"{split}.npz", link)
    empty = int((counts == 0).sum())
    print(f"train: {len(images)} tiles, {empty} empty ({empty / len(images):.0%}), "
          f"{len(new_tiles)} added, hard-negative score median {np.median(scores_kept):.3f}")


if __name__ == "__main__":
    main()
