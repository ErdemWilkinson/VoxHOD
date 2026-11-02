"""Export VoxHOD to full-int8 TFLite and score float and int8 models.

Two int8 files are scored when available:
  voxhod_int8.tflite      post-training quantization of the float model (written here)
  voxhod_qat_int8.tflite  the quantization-aware fine-tuned model (written by train.py)

Scoring follows the search-and-rescue goal: recall (share of annotated people
found) comes first. A prediction is a local maximum of the heatmap above a
threshold; it matches an unmatched person when its centre lies inside that
person's box grown by one grid cell. Reported per threshold, plus the best
F1 and the operating point with recall >= 0.90 if one exists.

    python scripts/export_and_evaluate.py [split ...]   (default: val test)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import ARTIFACTS, DATA, STRIDE, to_int8_tflite

THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.80, 0.05)]


def run_float(model, images):
    return model.predict(images.astype(np.float32) / 127.5 - 1.0, batch_size=64, verbose=0)


def run_int8(path: Path, images):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    (in_scale, in_zero), (out_scale, out_zero) = inp["quantization"], out["quantization"]
    outputs = []
    for image in images:
        x = np.round((image.astype(np.float32) / 127.5 - 1.0) / in_scale + in_zero)
        interpreter.set_tensor(inp["index"], np.clip(x, -128, 127).astype(np.int8)[None])
        interpreter.invoke()
        outputs.append((interpreter.get_tensor(out["index"])[0].astype(np.float32) - out_zero) * out_scale)
    return np.stack(outputs)


def peaks(heat: np.ndarray) -> np.ndarray:
    pooled = tf.nn.max_pool2d(heat[None, :, :, None], 3, 1, "SAME")[0, :, :, 0].numpy()
    return np.where(heat == pooled, heat, 0.0)


def score(outputs: np.ndarray, boxes: np.ndarray, counts: np.ndarray) -> dict:
    offsets = np.concatenate([[0], np.cumsum(counts)])
    found = {t: [0, 0, 0] for t in THRESHOLDS}  # true positives, predictions, people
    for tile, output in enumerate(outputs):
        gt = boxes[offsets[tile]:offsets[tile + 1]]
        local = peaks(output[..., 0])
        for threshold in THRESHOLDS:
            ys, xs = np.nonzero(local >= threshold)
            order = np.argsort(-local[ys, xs])
            centres = np.stack([(xs[order] + 0.5) * STRIDE, (ys[order] + 0.5) * STRIDE], 1)
            used = np.zeros(len(gt), bool)
            for cx, cy in centres:
                hit = np.flatnonzero(~used & (gt[:, 0] - STRIDE <= cx) & (cx <= gt[:, 2] + STRIDE)
                                     & (gt[:, 1] - STRIDE <= cy) & (cy <= gt[:, 3] + STRIDE))
                if len(hit):
                    used[hit[0]] = True
            found[threshold][0] += int(used.sum())
            found[threshold][1] += len(centres)
            found[threshold][2] += len(gt)
    table = {}
    for threshold, (tp, predicted, people) in found.items():
        recall, precision = tp / max(people, 1), tp / max(predicted, 1)
        table[threshold] = {"recall": recall, "precision": precision,
                            "f1": 2 * recall * precision / max(recall + precision, 1e-9),
                            "false_alarms_per_tile": (predicted - tp) / len(outputs)}
    best = max(table.items(), key=lambda item: item[1]["f1"])
    high_recall = [item for item in table.items() if item[1]["recall"] >= 0.90]
    return {"people": int(counts.sum()), "tiles": len(outputs), "by_threshold": table,
            "best_f1": {"threshold": best[0], **best[1]},
            "recall_90": ({"threshold": high_recall[-1][0], **high_recall[-1][1]} if high_recall else None)}


def main() -> None:
    splits = sys.argv[1:] or ["val", "test"]
    float_model = tf.keras.models.load_model(ARTIFACTS / "voxhod.keras", compile=False)
    variants = {"float": lambda images: run_float(float_model, images)}
    ptq = to_int8_tflite(float_model, "voxhod_int8.tflite")
    variants["int8_ptq"] = lambda images: run_int8(ptq, images)
    qat = ARTIFACTS / "voxhod_qat_int8.tflite"  # written by train.py after QAT fine-tuning
    if qat.exists():
        variants["int8_qat"] = lambda images: run_int8(qat, images)
    results = {"params": float_model.count_params()}
    for split in splits:
        data = np.load(DATA / f"{split}.npz")
        for variant, run in variants.items():
            result = score(run(data["images"]), data["boxes"], data["counts"])
            results[f"{split}/{variant}"] = result
            print(split, variant, "best F1", result["best_f1"], "recall>=0.9", result["recall_90"], flush=True)
    (ARTIFACTS / os.environ.get("VOX_EVAL_OUT", "evaluation.json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
