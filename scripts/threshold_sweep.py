"""Compare VoxHOD int8 files at matched operating points, with the threshold chosen on val.

A fixed threshold (0.2) is unfair between models whose scores are calibrated differently,
for example after training on many more empty tiles. This script:

  1. runs each int8 file once per split and caches the heatmaps (float16, 40x40),
  2. sweeps thresholds 0.02 ... 0.80 in steps of 0.01,
  3. for each model picks the highest threshold on **val** whose recall is not below a
     target recall, by default the reference model's val recall at 0.2,
  4. reports test and WiSARD recall, precision, false alarms per tile and recall of
     8-16 px people at that threshold, with a paired bootstrap (over tiles) of the
     difference to the reference model.

    python scripts/threshold_sweep.py <cache_dir> name=path.tflite [name=path.tflite ...]

The first model is the reference. Output: <cache_dir>/sweep.json and a printed table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

DATA = Path(os.environ.get("VOX_DATA", "D:/datasets/voxhod"))
STRIDE = 4
THRESHOLDS = np.round(np.arange(0.02, 0.805, 0.01), 2)
SPLITS = tuple(os.environ.get("VOX_SPLITS", "val,test,wisard").split(","))  # val must come first


def heatmaps(path: Path, images: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=os.cpu_count())
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    (si, zi), (so, zo) = inp["quantization"], out["quantization"]
    result = np.empty((len(images), 40, 40), np.float16)
    for i, image in enumerate(images):
        x = np.round((image.astype(np.float32) / 127.5 - 1.0) / si + zi)
        interpreter.set_tensor(inp["index"], np.clip(x, -128, 127).astype(np.int8)[None])
        interpreter.invoke()
        result[i] = (interpreter.get_tensor(out["index"])[0, ..., 0].astype(np.float32) - zo) * so
    return result


def peak_lists(heat: np.ndarray) -> list:
    """Per tile: (scores high to low, centres in px) of local maxima above the lowest threshold."""
    heat = heat.astype(np.float32)
    pooled = tf.nn.max_pool2d(heat[..., None], 3, 1, "SAME")[..., 0].numpy()
    lists = []
    for h, p in zip(heat, pooled):
        ys, xs = np.nonzero((h == p) & (h >= THRESHOLDS[0]))
        s = h[ys, xs]
        order = np.argsort(-s)
        lists.append((s[order], np.stack([(xs[order] + 0.5) * STRIDE, (ys[order] + 0.5) * STRIDE], 1)))
    return lists


def per_tile_counts(lists, boxes, counts):
    """[tiles, thresholds] arrays of true positives, predictions and small-people hits, plus per-tile people."""
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n, k = len(lists), len(THRESHOLDS)
    tp, pred, small_tp = (np.zeros((n, k), np.int32) for _ in range(3))
    small_people = np.zeros(n, np.int32)
    for t, (scores, centres) in enumerate(lists):
        gt = boxes[offsets[t]:offsets[t + 1]]
        height = gt[:, 3] - gt[:, 1]
        small = (height >= 8) & (height < 16)
        small_people[t] = int(small.sum())
        for j, threshold in enumerate(THRESHOLDS):
            keep = scores >= threshold
            pred[t, j] = int(keep.sum())
            if not len(gt) or not keep.any():
                continue
            used = np.zeros(len(gt), bool)
            for cx, cy in centres[keep]:
                hit = np.flatnonzero(~used & (gt[:, 0] - STRIDE <= cx) & (cx <= gt[:, 2] + STRIDE)
                                     & (gt[:, 1] - STRIDE <= cy) & (cy <= gt[:, 3] + STRIDE))
                if len(hit):
                    used[hit[0]] = True
            tp[t, j], small_tp[t, j] = int(used.sum()), int((used & small).sum())
    return tp, pred, small_tp, counts.astype(np.int32), small_people


def metrics(tp, pred, small_tp, people, small_people, idx=None):
    if idx is not None:
        tp, pred, small_tp, people, small_people = tp[idx], pred[idx], small_tp[idx], people[idx], small_people[idx]
    T, P, N = tp.sum(0), pred.sum(0), people.sum()
    recall, precision = T / max(N, 1), T / np.maximum(P, 1)
    return {"recall": recall, "precision": precision,
            "f1": 2 * recall * precision / np.maximum(recall + precision, 1e-9),
            "fa_per_tile": (P - T) / len(tp), "small_recall": small_tp.sum(0) / max(small_people.sum(), 1)}


def main() -> None:
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    models = dict(arg.split("=", 1) for arg in sys.argv[2:])
    names = list(models)
    stats, distance = {}, {}
    for split in SPLITS:
        data = np.load(DATA / f"{split}.npz")
        for name, path in models.items():
            file = cache / f"{name}_{split}.npy"
            if not file.exists():
                print(f"running {name} on {split} ({len(data['counts'])} tiles)", flush=True)
                np.save(file, heatmaps(Path(path), data["images"]))
            stats[name, split] = per_tile_counts(peak_lists(np.load(file)), data["boxes"], data["counts"])
        if "distance" in data.files:  # NOMAD: flight distance per tile
            distance[split] = data["distance"]
    target = float(os.environ.get("TARGET_RECALL", "nan"))
    if np.isnan(target):  # reference model's val recall at 0.20
        target = float(metrics(*stats[names[0], "val"])["recall"][list(THRESHOLDS).index(0.2)])
    chosen = {}
    for name in names:
        ok = np.flatnonzero(metrics(*stats[name, "val"])["recall"] >= target)
        chosen[name] = int(ok[-1]) if len(ok) else 0
    rng = np.random.default_rng(0)
    report = {"target_val_recall": target, "thresholds": THRESHOLDS.tolist(), "models": {}}
    print(f"\ntarget val recall {target:.3f} (threshold chosen on val, scored on test / wisard)")
    for name in names:
        j = chosen[name]
        entry = {"threshold": float(THRESHOLDS[j]), "curves": {}}
        for split in SPLITS:
            m = metrics(*stats[name, split])
            entry["curves"][split] = {k: v.tolist() for k, v in m.items()}
            entry[split] = {k: float(v[j]) for k, v in m.items()}
            if split in distance:
                tp, pred = stats[name, split][0][:, j], stats[name, split][1][:, j]
                entry[split]["fa_by_distance"] = {int(d): float((pred - tp)[distance[split] == d].mean())
                                                  for d in np.unique(distance[split])}
                print(f"{name:8s} {split} FA/tile by distance: " + ", ".join(
                    f"{d} m {v:.3f}" for d, v in entry[split]["fa_by_distance"].items()), flush=True)
            if name != names[0] and split != "val":
                ref, jr, cur = stats[names[0], split], chosen[names[0]], stats[name, split]
                n = len(cur[0])
                diffs = {"recall": [], "precision": [], "fa_per_tile": [], "small_recall": []}
                for _ in range(1000):
                    idx = rng.integers(0, n, n)
                    a, b = metrics(*cur, idx=idx), metrics(*ref, idx=idx)
                    for key in diffs:
                        diffs[key].append(a[key][j] - b[key][jr])
                entry[split]["vs_ref_ci95"] = {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                               for key, v in diffs.items()}
            e, ci = entry[split], entry[split].get("vs_ref_ci95")
            extra = ("  d_rec [{:+.3f},{:+.3f}] d_prec [{:+.3f},{:+.3f}] d_FA [{:+.3f},{:+.3f}]".format(
                *ci["recall"], *ci["precision"], *ci["fa_per_tile"]) if ci else "")
            print(f"{name:8s} t={THRESHOLDS[j]:.2f} {split:6s} rec {e['recall']:.3f} prec {e['precision']:.3f} "
                  f"FA {e['fa_per_tile']:.3f} F1 {e['f1']:.3f} small {e['small_recall']:.3f}{extra}", flush=True)
        report["models"][name] = entry
    (cache / "sweep.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
