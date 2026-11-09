"""Compare the device heatmaps (serial.log) with the PC TFLite reference (reference.npz).

    python compare.py serial.log reference.npz [threshold=0.2] [tiles]

tiles: use only the first N tiles. In the 2026-11-09 run, tiles from 163 on lay beyond the first
16 MB of flash, which this ESP-IDF build cannot read (3-byte flash addresses), so only 163 count.

For each model: share of heatmap cells that are bit-identical, largest difference in int8 steps,
tiles whose detections (3x3 local maxima above the threshold) differ, and recall / precision of
PC and device on the same tiles, scored as in scripts/threshold_sweep.py.
"""
import json
import re
import sys

import numpy as np

STRIDE = 4
log, ref_path = sys.argv[1], sys.argv[2]
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
ref = np.load(ref_path)
limit = int(sys.argv[4]) if len(sys.argv) > 4 else len(ref["counts"])
counts = ref["counts"][:limit]
offsets = np.concatenate([[0], np.cumsum(counts)])
boxes = ref["boxes"][:offsets[-1]]

device = {}
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.match(r"HEAT model=(\S+) i=(\d+) ([0-9a-f]{3200})$", line.strip())
    if m:
        device.setdefault(m[1], {})[int(m[2])] = np.frombuffer(bytes.fromhex(m[3]), np.int8).reshape(40, 40)


def peaks(h):
    p = np.pad(h, 1, constant_values=-np.inf)
    win = np.max([p[dy:dy + 40, dx:dx + 40] for dy in range(3) for dx in range(3)], 0)
    ys, xs = np.nonzero((h == win) & (h >= threshold))
    return set(zip(ys.tolist(), xs.tolist()))


def score(heats):
    found = pred = 0
    for t, h in enumerate(heats):
        pk = peaks(h); pred += len(pk)
        gt = boxes[offsets[t]:offsets[t + 1]]; used = np.zeros(len(gt), bool)
        for y, x in sorted(pk, key=lambda c: -h[c]):
            cx, cy = (x + 0.5) * STRIDE, (y + 0.5) * STRIDE
            hit = np.flatnonzero(~used & (gt[:, 0] - STRIDE <= cx) & (cx <= gt[:, 2] + STRIDE)
                                 & (gt[:, 1] - STRIDE <= cy) & (cy <= gt[:, 3] + STRIDE))
            if len(hit):
                used[hit[0]] = True
        found += int(used.sum())
    return {"recall": found / counts.sum(), "precision": found / max(pred, 1), "detections": pred}


result = {"tiles": len(counts), "people": int(counts.sum()), "threshold": threshold}
for name, dev in device.items():
    n = len(counts)
    if len(dev) < n:
        result[name] = {"error": f"{len(dev)} of {n} tiles on the device"}; continue
    pc_q, dev_q = ref[name][:n].astype(np.int16), np.stack([dev[i] for i in range(n)]).astype(np.int16)
    scale, zero = ref[name + "_out_q"]
    pc_f, dev_f = (pc_q - zero) * scale, (dev_q - zero) * scale
    diff = np.abs(pc_q - dev_q)
    result[name] = {"cells_identical": float((diff == 0).mean()), "max_diff_steps": int(diff.max()),
                    "cells_off_by_1": float((diff == 1).mean()),
                    "tiles_with_different_detections": int(sum(peaks(a) != peaks(b) for a, b in zip(pc_f, dev_f))),
                    "pc": score(pc_f), "device": score(dev_f)}
print(json.dumps(result, indent=1))
