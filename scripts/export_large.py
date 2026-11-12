"""Export VoxHOD Large (EfficientNet-B0 backbone, trained with VOX_ARCH=effb0) to a portable .h5 and a
full-int8 TFLite file, calibrated on the run's saved representative tiles (post-training quantization).

    VOX_ARCH=effb0 VOX_ARTIFACTS=<run dir> python scripts/export_large.py

Writes <run dir>/voxhod_large.h5 and <run dir>/voxhod_large_int8.tflite.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf  # noqa: E402

import train  # noqa: E402  (reads VOX_ARCH / VOX_ARTIFACTS from the environment)

model = tf.keras.models.load_model(Path(os.environ["VOX_ARTIFACTS"]) / "voxhod.keras", compile=False)
model.save(Path(os.environ["VOX_ARTIFACTS"]) / "voxhod_large.h5")
train.to_int8_tflite(model, "voxhod_large_int8.tflite")
