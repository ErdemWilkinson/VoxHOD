"""Average the train-split heatmaps of N float VoxHOD models for self-distillation.

    python scripts/ensemble_heat.py <data_dir> <out.npy> model1.keras model2.keras [model3.keras ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    data_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    model_paths = [Path(p) for p in sys.argv[3:]]
    images = np.load(data_dir / "train.npz")["images"]
    acc = np.zeros((len(images), 40, 40), dtype=np.float32)
    batch = 64
    for mp in model_paths:
        print(f"Predicting train heatmaps with {mp} ...", flush=True)
        model = tf.keras.models.load_model(mp, compile=False)
        for i in range(0, len(images), batch):
            x = images[i:i + batch].astype(np.float32) / 127.5 - 1.0
            acc[i:i + batch] += model(x, training=False).numpy()[..., 0]
        del model
        tf.keras.backend.clear_session()
    acc /= len(model_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, acc.astype(np.float16))
    print(f"Saved ensemble heatmap {acc.shape} (mean={acc.mean():.5f}, max={acc.max():.4f}) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
