"""Recall when the detector only ranks: the operator sees the top-K tiles of each frame.

A drone cannot raise every detection as an alarm (on new terrain that is about one false
alarm per tile). Used as a ranker, it sends the K highest-scoring tiles of each frame to a
person on the ground. What matters then is how many people end up in those K tiles.

Per frame (a HERIDAL test photo, ~35 tiles; a WiSARD video frame, 24 tiles), tiles are
ranked by their highest heatmap value. Reported for K = 1, 2, 3, 5:
  * people_recall: people inside the top-K tiles / all people
  * frame_hit: frames with people where at least one person is in the top-K tiles
  * random: the same measures for K tiles picked at random (expected value)
95% intervals: bootstrap over frames (2000). Model groups ("arms") are averaged over runs.

    python scripts/topk_recall.py <data_dir> <groups.npz> <out.json> name=cache_dir [name=cache_dir ...]

Heatmaps are read from <cache_dir>/<name>_<split>.npy (threshold_sweep.py cache).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KS = (1, 2, 3, 5)
SPLITS = {"test": "test_photo", "wisard": "wisard_frame"}


def per_frame_hits(tile_score, tile_people, frame):
    """[frames, len(KS)] people found in the top-K tiles, plus people per frame."""
    frames = frame.max() + 1
    found = np.zeros((frames, len(KS)))
    people = np.bincount(frame, weights=tile_people, minlength=frames)
    order = np.lexsort((-tile_score, frame))  # by frame, then score high to low
    start = np.searchsorted(frame[order], np.arange(frames))
    for f in range(frames):
        idx = order[start[f]:start[f + 1] if f + 1 < frames else len(order)]
        for j, k in enumerate(KS):
            found[f, j] = tile_people[idx[:k]].sum()
    return found, people


def summarise(found, people, weights):
    with_people = people > 0
    return {"people_recall": (weights @ found) / max(weights @ people, 1e-9),
            "frame_hit": (weights[with_people] @ (found[with_people] > 0)) / max(weights[with_people].sum(), 1e-9)}


def main() -> None:
    data, groups, out = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    models = dict(arg.split("=", 1) for arg in sys.argv[4:])
    rng = np.random.default_rng(0)
    report = {"K": list(KS), "splits": {}}
    for split, key in SPLITS.items():
        counts = np.load(data / f"{split}.npz")["counts"].astype(float)
        frame = groups[key].astype(int)
        frames = frame.max() + 1
        tiles_per_frame = np.bincount(frame, minlength=frames)
        people = np.bincount(frame, weights=counts, minlength=frames)
        # random K tiles: expected people found = people * K / tiles
        rnd_found = people[:, None] * np.minimum(np.array(KS)[None] / tiles_per_frame[:, None], 1.0)
        entry = {"frames": int(frames), "frames_with_people": int((people > 0).sum()), "people": int(people.sum()),
                 "random": {k: float(v) for k, v in zip(KS, rnd_found.sum(0) / people.sum())}, "models": {}}
        boots = rng.integers(0, frames, (2000, frames))
        for name, cache in models.items():
            file = Path(cache) / f"{name}_{split}.npy"
            if not file.exists():
                continue
            heat = np.load(file).astype(np.float32)
            score = heat.reshape(len(heat), -1).max(1)
            found, ppl = per_frame_hits(score, counts, frame)
            point = summarise(found, ppl, np.ones(frames))
            draws = {m: [] for m in point}
            for b in boots:
                s = summarise(found, ppl, np.bincount(b, minlength=frames).astype(float))
                for m in s:
                    draws[m].append(s[m])
            entry["models"][name] = {m: {str(k): [float(point[m][j]), float(np.percentile(np.array(draws[m])[:, j], 2.5)),
                                                  float(np.percentile(np.array(draws[m])[:, j], 97.5))]
                                         for j, k in enumerate(KS)} for m in point}
        report["splits"][split] = entry
        print(f"\n{split}: {entry['frames']} frames ({entry['frames_with_people']} with people, {entry['people']} people), "
              f"~{tiles_per_frame.mean():.0f} tiles per frame")
        print("  people in top-K tiles     " + "  ".join(f"K={k:<12d}" for k in KS))
        print(f"  {'random':24s}" + "  ".join(f"{entry['random'][k]:.3f}        " for k in KS))
        for name, m in entry["models"].items():
            print(f"  {name:24s}" + "  ".join(f"{m['people_recall'][str(k)][0]:.3f} [{m['people_recall'][str(k)][1]:.2f}-{m['people_recall'][str(k)][2]:.2f}]"
                                            for k in KS))
        print("  frames with a person found in top-K")
        for name, m in entry["models"].items():
            print(f"  {name:24s}" + "  ".join(f"{m['frame_hit'][str(k)][0]:.3f}        " for k in KS))
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
