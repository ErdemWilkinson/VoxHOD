"""Compare two groups of repeated training runs (arms) at their val-matched thresholds.

threshold_sweep.py compares every model with one reference model. With repeated runs of
two recipes the question is different: is the mean of arm B better than the mean of arm A?
Two kinds of uncertainty are reported:

  * run to run: the lowest and highest value among the runs of each arm,
  * tiles: a paired bootstrap over tiles (1000 resamples) of mean(B) - mean(A), each run
    scored at the threshold threshold_sweep.py chose for it on val.

The bootstrap does not include run-to-run variance; read both.

    python scripts/compare_arms.py <cache_dir> <data_dir> A=GA1,GA2,GA3 B=GB1,GB2,GB3 [splits]

<cache_dir> is the threshold_sweep.py cache (heatmaps and sweep.json). Output: <cache_dir>/arms.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import THRESHOLDS, metrics, peak_lists, per_tile_counts  # noqa: E402

KEYS = ("recall", "precision", "fa_per_tile")


def main() -> None:
    cache, data = Path(sys.argv[1]), Path(sys.argv[2])
    arms = {}
    for arg in sys.argv[3:5]:
        label, runs = arg.split("=", 1)
        arms[label] = runs.split(",")
    splits = sys.argv[5].split(",") if len(sys.argv) > 5 else ["test", "wisard", "nomad_tn_test"]
    sweep = json.loads((cache / "sweep.json").read_text())["models"]
    (a_label, a_runs), (b_label, b_runs) = arms.items()
    rng = np.random.default_rng(0)
    report = {"arms": arms, "splits": {}}
    for split in splits:
        d = np.load(data / f"{split}.npz")
        cols = {}
        for run in a_runs + b_runs:
            j = list(THRESHOLDS).index(round(sweep[run]["threshold"], 2))
            stats = per_tile_counts(peak_lists(np.load(cache / f"{run}_{split}.npy")), d["boxes"], d["counts"])
            cols[run] = tuple(s[:, j:j + 1] if s.ndim == 2 else s for s in stats)
        n = len(d["counts"])

        def arm_mean(runs, idx=None):
            values = [metrics(*cols[r], idx=idx) for r in runs]
            return {k: float(np.mean([v[k][0] for v in values])) for k in KEYS}

        per_run = {r: {k: float(metrics(*cols[r])[k][0]) for k in KEYS} for r in a_runs + b_runs}
        diffs = {k: [] for k in KEYS}
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            ma, mb = arm_mean(a_runs, idx), arm_mean(b_runs, idx)
            for k in KEYS:
                diffs[k].append(mb[k] - ma[k])
        entry = {"per_run": per_run, a_label: arm_mean(a_runs), b_label: arm_mean(b_runs),
                 "diff_ci95": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}}
        report["splits"][split] = entry
        print(f"\n{split} ({n} tiles)")
        for k in KEYS:
            ra = [per_run[r][k] for r in a_runs]
            rb = [per_run[r][k] for r in b_runs]
            lo, hi = entry["diff_ci95"][k]
            print(f"  {k:12s} {a_label} {entry[a_label][k]:.3f} [{min(ra):.3f}-{max(ra):.3f}]   "
                  f"{b_label} {entry[b_label][k]:.3f} [{min(rb):.3f}-{max(rb):.3f}]   "
                  f"{b_label}-{a_label} 95% [{lo:+.3f},{hi:+.3f}]", flush=True)
    # Analysis only (not a selection rule): put every run at the same HERIDAL test recall and
    # read false alarms and WiSARD recall at that threshold, so runs that landed at different
    # points of the same trade-off are compared like for like.
    level = float(__import__("os").environ.get("ARMS_TEST_RECALL", "0.78"))
    report["at_test_recall"] = {"level": level, "runs": {}}
    print(f"\nat HERIDAL test recall >= {level:.2f} (analysis only)")
    for run in a_runs + b_runs:
        curves = sweep[run]["curves"]
        ok = np.flatnonzero(np.array(curves["test"]["recall"]) >= level)
        j = int(ok[-1]) if len(ok) else 0
        row = {"threshold": float(THRESHOLDS[j]), "test_recall": curves["test"]["recall"][j],
               "test_fa": curves["test"]["fa_per_tile"][j]}
        for split in ("wisard", "nomad_tn_test"):
            if split in curves:
                row[f"{split}_recall"] = curves[split]["recall"][j]
                row[f"{split}_fa"] = curves[split]["fa_per_tile"][j]
        report["at_test_recall"]["runs"][run] = row
        print(f"  {run:5s} t={row['threshold']:.2f} test rec {row['test_recall']:.3f} FA {row['test_fa']:.3f}  "
              f"wisard rec {row.get('wisard_recall', float('nan')):.3f} FA {row.get('wisard_fa', float('nan')):.3f}  "
              f"nomad FA {row.get('nomad_tn_test_fa', float('nan')):.3f}", flush=True)
    (cache / "arms.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
