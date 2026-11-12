"""Run a float teacher (train.py with VOX_ARCH=effb0) over the train and val tiles once and save its
person heatmaps, so student runs can distil from them (VOX_TEACHER_HEAT) without running the teacher.

    python scripts/teacher_heat.py <teacher.keras> <data_dir> <out_dir>

Writes <out_dir>/train.npy and val.npy: float16 [tiles, 40, 40], tile order as in the .npz files.
The heatmaps are taken on the plain tiles; train.py turns and flips them together with the image.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    model_path, data, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    for split in os.environ.get("VOX_HEAT_SPLITS", "train,val").split(","):  # test splits: descriptive only
        images = np.load(data / f"{split}.npz")["images"]
        heat = np.empty((len(images), 40, 40), np.float16)
        for i in range(0, len(images), 64):
            x = images[i:i + 64].astype(np.float32) / 127.5 - 1.0
            heat[i:i + 64] = model(x, training=False).numpy()[..., 0]
        np.save(out / f"{split}.npy", heat)
        print(f"{split}: {len(images)} tiles, mean heat {float(heat.astype(np.float32).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
