"""Instant side-by-side comparison from cached heatmaps (no training):
   PUB_NANO (75 KB, 100% data) vs PUB_V3 (197 KB, 100% data) vs HSD / HSD2 / HSD3 / HSOUP (74.7 KB, 55% data)
   on strictly unseen splits: HERIDAL test, WiSARD, NOMAD (nomad_tn_test).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import THRESHOLDS, metrics, peak_lists, per_tile_counts  # noqa: E402

CACHE = Path("/mnt/d/vox_g")
DATA = Path("/mnt/d/datasets/voxhod_g")
MODELS = {
    "PUB_NANO (75KB,100%)": ("PUB_NANO", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_nano/voxhod_int8.tflite"),
    "PUB_V3 (197KB,100%)": ("PUB_V3", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_v3/voxhod_qat_int8.tflite"),
    "HSD1 (74.7KB,55%)": ("HSD", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_HSD_1123/voxhod_qat_int8.tflite"),
    "HSD2 (74.7KB,55%)": ("HSD2", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_HSD2_1123/voxhod_qat_int8.tflite"),
    "HSD3 (74.7KB,55%)": ("HSD3", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_HSD3_1123/voxhod_qat_int8.tflite"),
    "HSOUP (74.7KB,55%)": ("HSOUP", "/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_HSOUP_1123/voxhod_qat_int8.tflite"),
}


def load_curves(name: str, split: str) -> dict:
    d = np.load(DATA / f"{split}.npz")
    stats = per_tile_counts(peak_lists(np.load(CACHE / f"{name}_{split}.npy")), d["boxes"], d["counts"])
    return metrics(*stats)


def idx_at_recall(m: dict, target: float) -> int:
    ok = np.flatnonzero(m["recall"] >= target)
    return int(ok[-1]) if len(ok) else 0


def idx_at_fa(m: dict, max_fa: float) -> int:
    ok = np.flatnonzero(m["fa_per_tile"] <= max_fa)
    return int(ok[0]) if len(ok) else len(THRESHOLDS) - 1


def main() -> None:
    k20 = list(THRESHOLDS).index(0.20)
    rows = []
    print("=" * 112)
    print(f"{'Model':22s} {'Size':>7s} | {'t=0.20 Test (Rec/Prec/FA/Sm)':30s} | {'@80%Rec (FA/Prec/t/NOMAD)':27s} | {'@0.3FA Rec':10s} | {'WiSARD @0.5FA':13s}")
    print("-" * 112)
    for label, (key, tflite_path) in MODELS.items():
        sz_kb = Path(tflite_path).stat().st_size / 1024.0
        mt = load_curves(key, "test")
        mw = load_curves(key, "wisard")
        mn = load_curves(key, "nomad_tn_test")

        j80 = idx_at_recall(mt, 0.80)
        j03 = idx_at_fa(mt, 0.30)
        jw05 = idx_at_fa(mw, 0.50)

        t20_str = f"{mt['recall'][k20]*100:5.1f}% / {mt['precision'][k20]*100:4.1f}% / {mt['fa_per_tile'][k20]:.2f} / {mt['small_recall'][k20]*100:4.1f}%"
        r80_str = f"{mt['fa_per_tile'][j80]:.3f} / {mt['precision'][j80]*100:4.1f}% / t={THRESHOLDS[j80]:.2f} / {mn['fa_per_tile'][j80]:.2f}"
        f03_str = f"{mt['recall'][j03]*100:5.1f}% (s{mt['small_recall'][j03]*100:4.1f})"
        w05_str = f"{mw['recall'][jw05]*100:5.1f}% (P{mw['precision'][jw05]*100:4.1f})"

        print(f"{label:22s} {sz_kb:6.1f}K | {t20_str:30s} | {r80_str:27s} | {f03_str:10s} | {w05_str:13s}")
        rows.append({
            "label": label, "size_bytes": Path(tflite_path).stat().st_size,
            "test_t020": {"recall": float(mt["recall"][k20]), "precision": float(mt["precision"][k20]),
                          "fa": float(mt["fa_per_tile"][k20]), "small_recall": float(mt["small_recall"][k20])},
            "test_at_80rec": {"thr": float(THRESHOLDS[j80]), "fa": float(mt["fa_per_tile"][j80]),
                              "precision": float(mt["precision"][j80]), "nomad_fa": float(mn["fa_per_tile"][j80]),
                              "wisard_rec": float(mw["recall"][j80]), "wisard_fa": float(mw["fa_per_tile"][j80])},
            "test_at_03fa": {"thr": float(THRESHOLDS[j03]), "recall": float(mt["recall"][j03]),
                             "small_recall": float(mt["small_recall"][j03])},
            "wisard_at_05fa": {"thr": float(THRESHOLDS[jw05]), "recall": float(mw["recall"][jw05]),
                               "precision": float(mw["precision"][jw05])},
        })
    print("=" * 112)
    out = Path("/mnt/c/Users/erdem/OneDrive/Masaüstü/AI EĞİTİM/voxocr/artifacts_nano_v2_candidate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
