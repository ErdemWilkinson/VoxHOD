"""200 HERIDAL test tiles -> raw uint8 blob for the ESP32-P4 data partition, plus PC TFLite reference heatmaps.

150 tiles with people (every second one) and 50 empty tiles (evenly spaced) from the test split.
The blob is a 4-byte little-endian tile count followed by the 160x160x3 uint8 tiles. The device
quantizes each tile itself with the model's input scale and zero point, exactly as the PC does.

    python make_inputs.py <test.npz> <out_dir> name=model.tflite [name=model.tflite ...]

Writes <out_dir>/tiles.bin (flash it at 0x400000) and <out_dir>/reference.npz (not committed:
HERIDAL is not redistributed here).
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

data = np.load(sys.argv[1])
out = Path(sys.argv[2])
counts = data["counts"]
people, empty = np.flatnonzero(counts > 0), np.flatnonzero(counts == 0)
pick = np.sort(np.concatenate([people[::2][:150], empty[np.linspace(0, len(empty) - 1, 50).astype(int)]]))
offsets = np.concatenate([[0], np.cumsum(counts)])
images = data["images"][pick]
ref = {"index": pick, "counts": counts[pick],
       "boxes": np.concatenate([data["boxes"][offsets[i]:offsets[i + 1]] for i in pick])}
for arg in sys.argv[3:]:
    name, path = arg.split("=", 1)
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp, outd = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = inp["quantization"]
    heat = np.empty((len(images), 40, 40), np.int8)
    for k, image in enumerate(images):
        x = np.round((image.astype(np.float32) / 127.5 - 1.0) / si + zi)
        it.set_tensor(inp["index"], np.clip(x, -128, 127).astype(np.int8)[None]); it.invoke()
        heat[k] = it.get_tensor(outd["index"])[0, ..., 0]
    ref[name] = heat
    ref[name + "_out_q"] = np.array(outd["quantization"])
(out / "tiles.bin").write_bytes(np.uint32(len(images)).tobytes() + images.astype(np.uint8).tobytes())
np.savez(out / "reference.npz", **ref)
print(f"{len(images)} tiles ({int((ref['counts'] > 0).sum())} with people, {int(ref['counts'].sum())} people)")
