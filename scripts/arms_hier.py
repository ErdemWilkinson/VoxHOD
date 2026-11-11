"""Arm comparison with a hierarchical bootstrap (runs, then photos / frames) and two threshold rules.

Why: threshold_sweep.py / compare_arms.py resample single tiles and leave out run-to-run spread,
so their intervals are too narrow (tiles of one photo, and NOMAD tiles of one frame, are correlated).

Per run, two operating points chosen on val only:
  * recall rule  : highest threshold with val recall >= target (target = reference run's val recall at 0.20)
  * budget rule  : lowest threshold with val false alarms per tile <= budget (budget = reference's val FA at 0.20)
Curve measures on test (analysis, no selection): FA/tile at test recall R, recall at test FA/tile F.

Bootstrap (2000): resample runs within each arm with replacement; resample HERIDAL test photos and
NOMAD frames with replacement (shared by both arms, so the comparison stays paired). WiSARD is one clip:
reported per run only, no interval. Tile-level interval of the same difference shown for contrast.

    python arms_hier.py <cache_dir> <data_dir> <groups.npz> REF A=GA1,GA2,GA3 B=GB1,GB2,GB3 [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import THRESHOLDS, metrics, peak_lists, per_tile_counts  # noqa: E402

B = 2000
PAIRED = __import__("os").environ.get("ARMS_PAIRED") == "1"  # runs listed in the same order in A and B
REC_LEVELS = (0.75, 0.78, 0.80)
FA_LEVELS = (0.2, 0.3, 0.5)


def load_stats(cache, data, name, split):
    d = np.load(data / f"{split}.npz")
    tp, pred, _, people, _ = per_tile_counts(peak_lists(np.load(cache / f"{name}_{split}.npy")), d["boxes"], d["counts"])
    return tp.astype(np.float64), pred.astype(np.float64), people.astype(np.float64)


def grouped(stats, group):
    """Sum per-tile arrays into per-group arrays: tp[g,k], pred[g,k], people[g], tiles[g]."""
    tp, pred, people = stats
    n = group.max() + 1
    gtp = np.zeros((n, tp.shape[1])); np.add.at(gtp, group, tp)
    gpred = np.zeros_like(gtp); np.add.at(gpred, group, pred)
    return gtp, gpred, np.bincount(group, people, n), np.bincount(group, minlength=n).astype(np.float64)


def curve(g, w):
    """recall[k], fa_per_tile[k] for group weights w (bootstrap counts)."""
    tp, pred, people, tiles = w @ g[0], w @ g[1], w @ g[2], w @ g[3]
    return tp / max(people, 1e-9), (pred - tp) / max(tiles, 1e-9)


def fa_at_recall(rec, fa, r):
    ok = np.flatnonzero(rec >= r)  # recall falls as threshold rises
    return fa[ok[-1]] if len(ok) else np.nan


def recall_at_fa(rec, fa, f):
    ok = np.flatnonzero(fa <= f)  # fa falls as threshold rises
    return rec[ok[0]] if len(ok) else 0.0


def main():
    cache, data, groups = Path(sys.argv[1]), Path(sys.argv[2]), np.load(sys.argv[3])
    ref = sys.argv[4]
    arms = dict(a.split("=", 1) for a in sys.argv[5:7])
    arms = {k: v.split(",") for k, v in arms.items()}
    out_path = Path(sys.argv[7]) if len(sys.argv) > 7 else cache / "arms_hier.json"
    names = [ref] + [n for runs in arms.values() for n in runs if n != ref]
    k20 = list(THRESHOLDS).index(0.2)

    val = {n: load_stats(cache, data, n, "val") for n in names}
    rv = metrics(val[ref][0], val[ref][1], val[ref][0] * 0, val[ref][2], val[ref][2] * 0)
    target, budget = float(rv["recall"][k20]), float(rv["fa_per_tile"][k20])
    thr = {}
    for n in names:
        m = metrics(val[n][0], val[n][1], val[n][0] * 0, val[n][2], val[n][2] * 0)
        ok = np.flatnonzero(m["recall"] >= target)
        okb = np.flatnonzero(m["fa_per_tile"] <= budget)
        thr[n] = {"recall_rule": int(ok[-1]) if len(ok) else 0, "budget_rule": int(okb[0]) if len(okb) else len(THRESHOLDS) - 1}

    G = {"test": groups["test_photo"], "nomad_tn_test": groups["nomad_frame"], "wisard": groups["wisard_frame"]}
    g = {(n, s): grouped(load_stats(cache, data, n, s), G[s]) for n in names for s in G}
    tilestats = {(n, s): load_stats(cache, data, n, s) for n in names for s in ("test",)}

    report = {"target_val_recall": target, "val_fa_budget": budget, "runs": {}, "arms": {}, "diff": {}}
    for n in names:
        r = {"thr_recall_rule": float(THRESHOLDS[thr[n]["recall_rule"]]), "thr_budget_rule": float(THRESHOLDS[thr[n]["budget_rule"]])}
        for s in G:
            rec, fa = curve(g[n, s], np.ones(len(g[n, s][2])))
            for rule in ("recall_rule", "budget_rule"):
                j = thr[n][rule]
                r[f"{s}_{rule}"] = {"recall": float(rec[j]), "fa_per_tile": float(fa[j])}
            if s == "test":
                r["test_fa_at_recall"] = {str(x): float(fa_at_recall(rec, fa, x)) for x in REC_LEVELS}
                r["test_recall_at_fa"] = {str(x): float(recall_at_fa(rec, fa, x)) for x in FA_LEVELS}
        report["runs"][n] = r

    rng = np.random.default_rng(0)
    (la, ra), (lb, rb) = arms.items()
    measures = {}

    def arm_values(runs, wt, wn):
        vals = {}
        for rule in ("recall_rule", "budget_rule"):
            for s, w in (("test", wt), ("nomad_tn_test", wn)):
                rec_fa = [curve(g[n, s], w) for n in runs]
                vals[f"{s}_{rule}_recall"] = np.mean([rf[0][thr[n][rule]] for n, rf in zip(runs, rec_fa)])
                vals[f"{s}_{rule}_fa"] = np.mean([rf[1][thr[n][rule]] for n, rf in zip(runs, rec_fa)])
        tc = [curve(g[n, "test"], wt) for n in runs]
        for x in REC_LEVELS:
            vals[f"test_fa_at_recall_{x}"] = np.nanmean([fa_at_recall(r, f, x) for r, f in tc])
        for x in FA_LEVELS:
            vals[f"test_recall_at_fa_{x}"] = np.mean([recall_at_fa(r, f, x) for r, f in tc])
        return vals

    npho, nfr = len(g[ra[0], "test"][2]), len(g[ra[0], "nomad_tn_test"][2])
    point_a = arm_values(ra, np.ones(npho), np.ones(nfr))
    point_b = arm_values(rb, np.ones(npho), np.ones(nfr))
    for _ in range(B):
        wt = np.bincount(rng.integers(0, npho, npho), minlength=npho).astype(float)
        wn = np.bincount(rng.integers(0, nfr, nfr), minlength=nfr).astype(float)
        if PAIRED:  # arm B run i comes from the same training run as arm A run i: resample pairs
            pick = rng.integers(0, len(ra), len(ra))
            sa, sb = [ra[i] for i in pick], [rb[i] for i in pick]
        else:
            sa = list(rng.choice(ra, len(ra))); sb = list(rng.choice(rb, len(rb)))
        va, vb = arm_values(sa, wt, wn), arm_values(sb, wt, wn)
        for key in va:
            measures.setdefault(key, []).append(vb[key] - va[key])
    for key in point_a:
        d = np.array(measures[key])
        report["arms"][key] = {la: float(point_a[key]), lb: float(point_b[key])}
        report["diff"][key] = {"point": float(point_b[key] - point_a[key]),
                               "hier_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}

    # tile-level interval (old method, runs fixed) for the recall-rule test differences, for contrast
    nt = len(tilestats[ra[0], "test"][0])
    tdiff = {"recall": [], "fa": []}
    for _ in range(1000):
        idx = rng.integers(0, nt, nt)

        def armmean(runs):
            rr, ff = [], []
            for n in runs:
                tp, pred, people = (x[idx] for x in tilestats[n, "test"])
                j = thr[n]["recall_rule"]
                rr.append(tp[:, j].sum() / people.sum()); ff.append((pred[:, j] - tp[:, j]).sum() / nt)
            return np.mean(rr), np.mean(ff)
        a, b = armmean(ra), armmean(rb)
        tdiff["recall"].append(b[0] - a[0]); tdiff["fa"].append(b[1] - a[1])
    report["tile_level_ci95_test_recall_rule"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in tdiff.items()}
    out_path.write_text(json.dumps(report, indent=1))

    print(f"target val recall {target:.4f}, val FA budget {budget:.3f}/tile (reference {ref} at 0.20)")
    print(f"{'run':6s} thrR  thrB | test R-rule rec/FA | test B-rule rec/FA | NOMAD FA R/B | WiSARD rec R/B")
    for n in names:
        r = report["runs"][n]
        print(f"{n:6s} {r['thr_recall_rule']:.2f}  {r['thr_budget_rule']:.2f} | "
              f"{r['test_recall_rule']['recall']:.3f} {r['test_recall_rule']['fa_per_tile']:.3f} | "
              f"{r['test_budget_rule']['recall']:.3f} {r['test_budget_rule']['fa_per_tile']:.3f} | "
              f"{r['nomad_tn_test_recall_rule']['fa_per_tile']:.2f} {r['nomad_tn_test_budget_rule']['fa_per_tile']:.2f} | "
              f"{r['wisard_recall_rule']['recall']:.3f} {r['wisard_budget_rule']['recall']:.3f}")
    print(f"\n{lb} - {la} (mean of runs; hierarchical 95% CI: runs + photos/frames)")
    for key, v in report["diff"].items():
        print(f"  {key:34s} {la} {report['arms'][key][la]:.3f}  {lb} {report['arms'][key][lb]:.3f}  "
              f"diff {v['point']:+.3f}  [{v['hier_ci95'][0]:+.3f}, {v['hier_ci95'][1]:+.3f}]")
    t = report["tile_level_ci95_test_recall_rule"]
    print(f"  (old tile-level CI, recall rule, test: recall [{t['recall'][0]:+.3f},{t['recall'][1]:+.3f}], FA [{t['fa'][0]:+.3f},{t['fa'][1]:+.3f}])")


if __name__ == "__main__":
    main()
