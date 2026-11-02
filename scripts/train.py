"""Train VoxHOD: a tiny int8-friendly person detector for drone search and rescue.

CenterNet-style, anchor-free (PP-PicoDet is anchor-free too): the network
predicts a 40x40 person-centre heatmap (stride 4 on a 160x160 RGB tile) plus
the box width/height at each centre. Only Conv2D, DepthwiseConv2D, batch norm,
ReLU, Add and nearest upsampling are used, all of which quantize cleanly to
int8 and exist in TFLite Micro.

Environment:
  VOX_DATA         folder with train.npz / val.npz from prepare_heridal.py
  VOX_ARTIFACTS    output folder (default ../artifacts)
  VOX_MAX_MINUTES  float training budget, cosine learning-rate decay over it
  VOX_QAT_MINUTES  quantization-aware fine-tuning budget after float training (0 = skip)
  VOX_THREADS      CPU threads
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VOX_DATA", "D:/datasets/voxhod"))
ARTIFACTS = Path(os.environ.get("VOX_ARTIFACTS", ROOT / "artifacts"))
TILE, STRIDE = 160, 4
GRID = TILE // STRIDE
WIDTH = float(os.environ.get("VOX_WIDTH", "1"))  # filter multiplier for smaller models
if os.environ.get("VOX_THREADS"):
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["VOX_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(1)


def conv_bn(x, filters, kernel=3, stride=1):
    x = tf.keras.layers.Conv2D(filters, kernel, stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def separable(x, filters, stride=1):
    """Depthwise 3x3 + pointwise 1x1, residual when the shape allows."""
    y = tf.keras.layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = conv_bn(y, filters, 1)
    if stride == 1 and x.shape[-1] == filters:
        y = tf.keras.layers.Add()([x, y])
    return y


def build_model() -> tf.keras.Model:
    w = lambda filters: int(filters * WIDTH)
    image = tf.keras.Input((TILE, TILE, 3), name="image")
    x = conv_bn(image, w(16), 3, 2)                       # 80
    x = separable(x, w(32))
    c4 = separable(separable(x, w(48), 2), w(48))            # 40
    c8 = separable(separable(c4, w(96), 2), w(96))           # 20
    c16 = separable(separable(c8, w(160), 2), w(160))        # 10
    p = conv_bn(c16, w(64), 1)
    p = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D()(p), conv_bn(c8, w(64), 1)])
    p = separable(p, w(64))                               # 20
    p = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D()(p), conv_bn(c4, w(64), 1)])
    p = separable(p, w(64))                               # 40
    p = conv_bn(p, w(64), 3)
    heat = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="heat",
                                  bias_initializer=tf.keras.initializers.Constant(-2.19))(p)
    size = tf.keras.layers.Conv2D(2, 1, name="size")(p)
    return tf.keras.Model(image, tf.keras.layers.Concatenate(name="detections")([heat, size]))


def encode_targets(boxes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per tile: [heat, log_w, log_h, centre_mask] on the 40x40 grid. Heat is a
    Gaussian around each centre (sigma from box size, as in CenterNet) with
    exactly 1.0 at the centre cell; sizes are log of grid units."""
    targets = np.zeros((len(counts), GRID, GRID, 4), np.float32)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    start = 0
    for tile, count in enumerate(counts):
        for x0, y0, x1, y1 in boxes[start:start + count]:
            w, h = max(x1 - x0, 1.0) / STRIDE, max(y1 - y0, 1.0) / STRIDE
            cx, cy = min(int((x0 + x1) / 2 / STRIDE), GRID - 1), min(int((y0 + y1) / 2 / STRIDE), GRID - 1)
            sigma = max(0.5, min(w, h) / 3)
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            targets[tile, :, :, 0] = np.maximum(targets[tile, :, :, 0], gaussian)
            targets[tile, cy, cx] = [1.0, np.log(w), np.log(h), 1.0]
        start += count
    return targets


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA / f"{name}.npz")
    return data["images"], encode_targets(data["boxes"], data["counts"])


def detection_loss(target, output):
    heat, mask = target[..., 0], target[..., 3]
    pred = tf.clip_by_value(output[..., 0], 1e-4, 1 - 1e-4)
    positive = tf.cast(tf.equal(heat, 1.0), tf.float32)
    focal = -(positive * (1 - pred) ** 2 * tf.math.log(pred)
              + (1 - positive) * (1 - heat) ** 4 * pred ** 2 * tf.math.log(1 - pred))
    count = tf.maximum(tf.reduce_sum(positive), 1.0)
    size = tf.reduce_sum(mask[..., None] * tf.abs(output[..., 1:3] - target[..., 1:3]))
    return tf.reduce_sum(focal) / count + 0.1 * size / count


def augment(image, target):
    """Flips and 90-degree turns (a nadir camera has no 'up'), applied to the
    image and the target grid alike; turning swaps width and height. Then
    colour, contrast, haze, blur and sensor noise for weather, light and
    drone vibration."""
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    if tf.random.uniform([]) < 0.5:
        image, target = tf.image.flip_left_right(image), tf.image.flip_left_right(target)
    if tf.random.uniform([]) < 0.5:
        image, target = tf.image.flip_up_down(image), tf.image.flip_up_down(target)
    if tf.random.uniform([]) < 0.5:
        image, target = tf.image.rot90(image), tf.image.rot90(target)
        target = tf.stack([target[..., 0], target[..., 2], target[..., 1], target[..., 3]], -1)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    if tf.random.uniform([]) < 0.2:  # haze / fog
        image = image * 0.6 + 0.4 * tf.random.uniform([], 0.3, 0.9)
    if tf.random.uniform([]) < 0.25:  # motion / focus blur
        image = tf.nn.avg_pool2d(image[None], 3, 1, "SAME")[0]
    image += tf.random.normal(tf.shape(image), stddev=tf.random.uniform([], 0.0, 0.05))
    return tf.clip_by_value(image, -1.0, 1.0), target


def dataset(images, targets, train: bool) -> tf.data.Dataset:
    data = tf.data.Dataset.from_tensor_slices((images, targets))
    if train:
        data = data.shuffle(len(images)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        data = data.map(lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, y))
    return data.batch(32).prefetch(tf.data.AUTOTUNE)


class TimeCosine(tf.keras.callbacks.Callback):
    """Cosine learning-rate decay over a time budget, stop when the next epoch
    would not fit. With VOX_KEEP=best the best val_loss weights are restored;
    the default keeps the final (low learning rate) weights, because val tiles
    are mostly empty terrain while train tiles are mostly people, so val_loss
    swings and picked an epoch-1 model in the first run."""

    def __init__(self, base_lr: float, minutes: float):
        super().__init__()
        self.base_lr, self.seconds = base_lr, minutes * 60.0

    def on_train_begin(self, logs=None):
        self.started, self.best, self.weights = time.monotonic(), float("inf"), None

    def on_epoch_begin(self, epoch, logs=None):
        progress = min((time.monotonic() - self.started) / self.seconds, 1.0)
        lr = self.base_lr * (0.3 if epoch == 0 else 0.5 * (1 + np.cos(np.pi * progress)))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, max(lr, 1e-6))

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.weights = logs["val_loss"], self.model.get_weights()
        elapsed = time.monotonic() - self.started
        if elapsed + elapsed / (epoch + 1) > self.seconds:
            self.model.stop_training = True
            print(f"Budget reached after epoch {epoch + 1}; best val_loss {self.best:.4f}", flush=True)

    def on_train_end(self, logs=None):
        if self.weights is not None and os.environ.get("VOX_KEEP") == "best":
            self.model.set_weights(self.weights)


def to_int8_tflite(model: tf.keras.Model, name: str) -> Path:
    """Full-int8 TFLite (weights, activations, input, output) calibrated on
    the saved representative tiles."""
    samples = np.load(ARTIFACTS / "representative_images.npy").astype(np.float32) / 127.5 - 1.0
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None]] for sample in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path = ARTIFACTS / name
    path.write_bytes(converter.convert())
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)", flush=True)
    return path


def fit(model, train, val, lr, minutes):
    model.compile(tf.keras.optimizers.Adam(lr, clipnorm=5.0), loss=detection_loss)
    return model.fit(train, validation_data=val, epochs=1000, verbose=2,
                     callbacks=[TimeCosine(lr, minutes)]).history


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train_images, train_targets = load_split("train")
    val_images, val_targets = load_split("val")
    print(f"Train tiles: {len(train_images)}, validation tiles: {len(val_images)}", flush=True)
    train, val = dataset(train_images, train_targets, True), dataset(val_images, val_targets, False)

    model = build_model()
    if os.environ.get("VOX_WARM_START"):  # continue from an earlier voxhod.keras
        model.set_weights(tf.keras.models.load_model(os.environ["VOX_WARM_START"], compile=False).get_weights())
        print(f"Warm start from {os.environ['VOX_WARM_START']}", flush=True)
    print(f"VoxHOD parameters: {model.count_params():,}", flush=True)
    history = fit(model, train, val, 2e-3, float(os.environ.get("VOX_MAX_MINUTES", 60)))
    model.save(ARTIFACTS / "voxhod.keras")
    (ARTIFACTS / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in history.items()}))
    calibration = np.random.default_rng(7).choice(len(train_images), min(300, len(train_images)), replace=False)
    np.save(ARTIFACTS / "representative_images.npy", train_images[calibration])

    qat_minutes = float(os.environ.get("VOX_QAT_MINUTES", 0))
    if qat_minutes > 0:
        try:
            import tensorflow_model_optimization as tfmot
            qat_model = tfmot.quantization.keras.quantize_model(model)
            qat_history = fit(qat_model, train, val, 2e-4, qat_minutes)
            # A saved QAT model does not reload cleanly (quantize_layer
            # variables), so convert it while it is still in memory.
            to_int8_tflite(qat_model, "voxhod_qat_int8.tflite")
            (ARTIFACTS / "history_qat.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in qat_history.items()}))
            print("QAT fine-tuning done.", flush=True)
        except Exception as error:  # QAT is an improvement, not a requirement: keep the float model
            print(f"QAT skipped: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
