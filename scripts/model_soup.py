"""Model Soup: average float weights of N models sharing a common init, recalibrate BatchNorm, save.

    python scripts/model_soup.py <data_dir> <out_model.keras> model1.keras model2.keras [model3.keras ...]
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
    models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]
    weights = [m.get_weights() for m in models]
    avg_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]
    soup = models[0]
    soup.set_weights(avg_weights)
    for m in models[1:]:
        del m

    # Reset and recalibrate BatchNorm running mean/variance on train tiles
    images = np.load(data_dir / "train.npz")["images"]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(images))[:min(4096, len(images))]
    for layer in soup.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.0
    batch = 32
    seen = 0
    for i in range(0, len(idx), batch):
        x = images[idx[i:i + batch]].astype(np.float32) / 127.5 - 1.0
        # Progressive momentum = seen / (seen + len(x)) computes exact running mean/var
        mom = seen / (seen + len(x))
        for layer in soup.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = mom
        soup(x, training=True)
        seen += len(x)
    for layer in soup.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.99

    out_path.parent.mkdir(parents=True, exist_ok=True)
    soup.save(out_path)
    print(f"Saved souped model ({len(model_paths)} models, BN recalibrated on {seen} tiles) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
