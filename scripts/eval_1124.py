"""Comprehensive unseen-split benchmark for 2026-11-24 overnight marathon:
   Scores published nano & v3 alongside HSD_SOUP, FN_HSD, FN_SOUP (74.7 KB) and FV_HSD, FV_SOUP (~194 KB)
   on strictly unseen splits: HERIDAL test (3507 tiles), WiSARD (6336 tiles), NOMAD full (4000 tiles).
   Also evaluates 2-of-3 frame temporal persistence on WiSARD (0 extra KB).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import THRESHOLDS, heatmaps, metrics, peak_lists, per_tile_counts  # noqa: E402

WIN = Path("/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr")
DATA = Path("/mnt/d/datasets/voxhod")
CACHE = Path("/mnt/d/vox_1124")
CACHE.mkdir(parents=True, exist_ok=True)
OUT = WIN / "artifacts_night_1124"

MODELS = [
    ("PUB_NANO (74K,100%)", "PUB_NANO", WIN / "artifacts_nano/voxhod_int8.tflite"),
    ("PUB_V3 (192K,100%)", "PUB_V3", WIN / "artifacts_v3/voxhod_qat_int8.tflite"),
    ("HSD1 (75K,55%)", "HSD1", WIN / "artifacts_HSD_1123/voxhod_qat_int8.tflite"),
    ("HSOUP (75K,55%)", "HSOUP", WIN / "artifacts_HSOUP_1123/voxhod_qat_int8.tflite"),
    ("HSD_SOUP (75K,55%)", "HSD_SOUP", WIN / "artifacts_HSD_SOUP_1124/voxhod_qat_int8.tflite"),
    ("FN_HSD (75K,100%)", "FN_HSD", WIN / "artifacts_FN_HSD_1124/voxhod_qat_int8.tflite"),
    ("FN_SOUP (75K,100%)", "FN_SOUP", WIN / "artifacts_FN_SOUP_1124/voxhod_qat_int8.tflite"),
    ("FV_HSD (194K,100%)", "FV_HSD", WIN / "artifacts_FV_HSD_1124/voxhod_qat_int8.tflite"),
    ("FV_SOUP (194K,100%)", "FV_SOUP", WIN / "artifacts_FV_SOUP_1124/voxhod_qat_int8.tflite"),
]
SPLITS = ("test", "wisard", "nomad_tn")


def idx_at_recall(m: dict, target: float) -> int:
    ok = np.flatnonzero(m["recall"] >= target)
    return int(ok[-1]) if len(ok) else 0


def idx_at_fa(m: dict, max_fa: float) -> int:
    ok = np.flatnonzero(m["fa_per_tile"] <= max_fa)
    return int(ok[0]) if len(ok) else len(THRESHOLDS) - 1


def main() -> None:
    datasets = {s: np.load(DATA / f"{s}.npz") for s in SPLITS}
    k20 = list(THRESHOLDS).index(0.20)
    rows = []

    print("=" * 124)
    print(f"{'Model':21s} {'Size':>6s} | {'t=0.20 Test (Rec/Prec/FA/Sm)':29s} | {'@80%Rec (FA/Prec/t/NOMAD4k)':28s} | {'@0.3FA Rec(Sm)':14s} | {'WiSARD @0.5FA':14s}")
    print("-" * 124)

    for label, key, tflite_path in MODELS:
        if not tflite_path.exists():
            print(f"{label:21s} MISSING ({tflite_path})")
            continue
        sz_kb = tflite_path.stat().st_size / 1024.0
        cur = {}
        for s in SPLITS:
            cf = CACHE / f"{key}_{s}.npy"
            if not cf.exists():
                np.save(cf, heatmaps(tflite_path, datasets[s]["images"]))
            hm = np.load(cf)
            stats = per_tile_counts(peak_lists(hm), datasets[s]["boxes"], datasets[s]["counts"])
            cur[s] = metrics(*stats)

        mt, mw, mn = cur["test"], cur["wisard"], cur["nomad_tn"]
        j80 = idx_at_recall(mt, 0.80)
        j03 = idx_at_fa(mt, 0.30)
        jw05 = idx_at_fa(mw, 0.50)

        t20 = f"{mt['recall'][k20]*100:5.1f}%/{mt['precision'][k20]*100:4.1f}%/{mt['fa_per_tile'][k20]:.2f}/{mt['small_recall'][k20]*100:4.1f}%"
        r80 = f"{mt['fa_per_tile'][j80]:.3f}/{mt['precision'][j80]*100:4.1f}%/t={THRESHOLDS[j80]:.2f}/{mn['fa_per_tile'][j80]:.2f}"
        f03 = f"{mt['recall'][j03]*100:5.1f}% ({mt['small_recall'][j03]*100:4.1f}%)"
        w05 = f"{mw['recall'][jw05]*100:5.1f}% (P{mw['precision'][jw05]*100:4.1f})"
        print(f"{label:21s} {sz_kb:5.1f}K | {t20:29s} | {r80:28s} | {f03:14s} | {w05:14s}")

        rows.append({
            "label": label, "key": key, "size_bytes": tflite_path.stat().st_size,
            "test_t020": {"recall": float(mt["recall"][k20]), "precision": float(mt["precision"][k20]),
                          "fa": float(mt["fa_per_tile"][k20]), "small_recall": float(mt["small_recall"][k20])},
            "test_at_80rec": {"thr": float(THRESHOLDS[j80]), "fa": float(mt["fa_per_tile"][j80]),
                              "precision": float(mt["precision"][j80]), "nomad4k_fa": float(mn["fa_per_tile"][j80]),
                              "wisard_rec": float(mw["recall"][j80]), "wisard_fa": float(mw["fa_per_tile"][j80])},
            "test_at_03fa": {"thr": float(THRESHOLDS[j03]), "recall": float(mt["recall"][j03]),
                             "small_recall": float(mt["small_recall"][j03])},
            "wisard_at_05fa": {"thr": float(THRESHOLDS[jw05]), "recall": float(mw["recall"][jw05]),
                               "precision": float(mw["precision"][jw05])},
        })
    print("=" * 124)

    # Bonus: WiSARD 2-of-3 frame temporal persistence (same tile slot across consecutive frames, 24 tiles/frame)
    print("\n=== WiSARD Temporal Persistence (2-of-3 consecutive frames at t=0.20, 0 extra KB) ===")
    w_boxes, w_counts = datasets["wisard"]["boxes"], datasets["wisard"]["counts"]
    n_tiles = len(w_counts)
    tpf = 24  # 264 frames * 24 tiles = 6336
    if n_tiles % tpf == 0:
        n_frames = n_tiles // tpf
        for label, key, tflite_path in MODELS:
            cf = CACHE / f"{key}_wisard.npy"
            if not cf.exists():
                continue
            hm = np.load(cf).reshape(n_frames, tpf, 40, 40)
            # Temporal persistence: tile-level score smoothed by median of 3 consecutive frames (2-of-3 logic)
            # For each frame f, a pixel is active if >=2 of {f-1, f, f+1} exceed threshold (approx via 3-frame median)
            padded = np.pad(hm, ((1, 1), (0, 0), (0, 0), (0, 0)), mode="edge")
            med = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0), axis=0).reshape(n_tiles, 40, 40)
            m_raw = metrics(*per_tile_counts(peak_lists(hm.reshape(n_tiles, 40, 40)), w_boxes, w_counts))
            m_tmp = metrics(*per_tile_counts(peak_lists(med), w_boxes, w_counts))
            print(f"{label:21s} RAW t=0.20: rec {m_raw['recall'][k20]*100:5.1f}% FA {m_raw['fa_per_tile'][k20]:.3f}  ->  "
                  f"3-FRAME MEDIAN: rec {m_tmp['recall'][k20]*100:5.1f}% FA {m_tmp['fa_per_tile'][k20]:.3f} "
                  f"(FA {(m_tmp['fa_per_tile'][k20]/max(m_raw['fa_per_tile'][k20],1e-6)-1)*100:+.0f}%)")

    (OUT / "benchmark_1124.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
