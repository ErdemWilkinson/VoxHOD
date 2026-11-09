"""Draw the VoxHOD README charts into docs/ (light and dark SVG).

Numbers at the fixed threshold 0.2, int8 files (v3 QAT, nano PTQ):
artifacts_v3/evaluation.json, artifacts_v3/evaluation_wisard.json,
artifacts_nano/evaluation.json. The diagnosis numbers (v3, HERIDAL test set,
threshold 0.2, 3x3 peaks) come from a separate analysis run on 2026-11-07. The
experiment, NOMAD and pruning numbers come from scripts/threshold_sweep.py
(2026-11-09, threshold chosen on validation; NOMAD at the fixed threshold 0.2).

    python scripts/make_readme_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import grouped_bars, hbar_panels, roadmap  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    grouped_bars(DOCS / "voxhod-models", "Published models at threshold 0.2, int8 files",
                 "HERIDAL test set (3,507 tiles, 372 people) and one WiSARD clip never trained on (6,336 tiles, 1,271 people).",
                 ["HERIDAL recall", "HERIDAL precision", "WiSARD recall", "WiSARD precision"],
                 [("v3 · 197 KB", 0, [83.9, 20.4, 61.3, 19.1]), ("nano · 75 KB", 1, [81.2, 12.7, 62.2, 13.4])])
    hbar_panels(DOCS / "voxhod-diagnosis", "Where the false alarms come from (v3, HERIDAL test, threshold 0.2)",
                "The model saw few empty tiles in training, but most tiles in real use are empty.",
                [{"title": "False alarms", "max": 1200, "slot": 0,
                  "rows": [("on empty tiles", 1087, "1,087"), ("on tiles with people", 134, "134")]},
                 {"title": "Share of empty tiles", "max": 100, "slot": 0,
                  "rows": [("training", 28.5, "29%"), ("validation", 96, "96%"), ("test", 91, "91%")]},
                 {"title": "People found, by size", "max": 100, "slot": 0,
                  "rows": [("8-16 px (171)", 77.2, "77.2%"), ("16-32 px (198)", 89.9, "89.9%")]}])
    grouped_bars(DOCS / "voxhod-experiments", "Experiments at matched operating points (int8 QAT files, 197 KB)",
                 "Threshold chosen on validation to find 86.8% of people there, like v3 at 0.2; scored on HERIDAL test and WiSARD.",
                 ["HERIDAL recall", "HERIDAL precision", "WiSARD recall", "WiSARD precision"],
                 [("v3", 0, [83.9, 20.4, 61.3, 19.1]), ("HN, +1 h", 1, [81.5, 26.1, 57.5, 24.4]),
                  ("control, +1 h", 4, [79.8, 26.3, 56.5, 25.3]), ("PW15, +30 min", 3, [80.6, 25.3, 57.0, 25.0]),
                  ("control, +30 min", 2, [82.8, 21.5, 59.6, 19.9])])
    grouped_bars(DOCS / "voxhod-nomad", "False alarms per empty tile at threshold 0.2 (int8 files)",
                 "HERIDAL test (3,207 empty tiles) against NOMAD, another camera and terrain never trained on (800 empty tiles per height).",
                 ["HERIDAL", "NOMAD 10 m", "30 m", "50 m", "70 m", "90 m"],
                 [("v3 · 197 KB", 0, [0.34, 1.10, 1.11, 1.39, 1.35, 1.49]),
                  ("nano · 75 KB", 1, [0.60, 1.34, 1.25, 1.47, 1.54, 1.56])],
                 maximum=2.0, unit="", ticks=(0, 0.5, 1.0, 1.5, 2.0))
    hbar_panels(DOCS / "voxhod-pruning", "Small models: pruned from v3 against trained from scratch (int8 QAT files)",
                "Precision at a fixed recall, read off each threshold curve. Width 0.5 was run three times per recipe.",
                [{"title": "HERIDAL precision at 80% recall", "max": 30, "slot": 0,
                  "rows": [("v3 · 197 KB", 24.3, "24.3%"), ("pruned 0.7", 24.5, "24.5%"), ("scratch 0.7", 15.0, "15.0%"),
                           ("nano · 75 KB", 14.7, "14.7%"), ("pruned 0.5, run 1", 18.6, "18.6%"), ("run 2", 20.2, "20.2%"),
                           ("run 3", 17.6, "17.6%"), ("scratch 0.5, run 1", 14.7, "14.7%"), ("run 2", 27.0, "27.0%"),
                           ("run 3", 16.0, "16.0%")]},
                 {"title": "WiSARD precision at 60% recall", "max": 30, "slot": 2,
                  "rows": [("v3 · 197 KB", 19.1, "19.1%"), ("pruned 0.7", 26.3, "26.3%"), ("scratch 0.7", 20.2, "20.2%"),
                           ("nano · 75 KB", 15.6, "15.6%"), ("pruned 0.5, run 1", 22.2, "22.2%"), ("run 2", 23.7, "23.7%"),
                           ("run 3", 21.5, "21.5%"), ("scratch 0.5, run 1", 22.4, "22.4%"), ("run 2", 22.0, "22.0%"),
                           ("run 3", 23.9, "23.9%")]}])
    roadmap(DOCS / "roadmap-voxhod", "VoxHOD roadmap",
            "Rule: the device model never grows; larger models are only teachers. A step ships only if it passes its gate.",
            ["Next", "This week", "Later"],
            [("VoxHOD", [[("A larger validation split", "Matched on 317 validation people, models still land between 77% and 87% test recall; fix this before more comparisons", True),
                          ("Fewer false alarms, same recall", "Tried 2026-11-08/09: hard negatives, 1.5x weight, extra training, pruning and a QAT nano all slid along the same trade-off. Next: a larger teacher", False)],
                         [("More varied empty terrain", "NOMAD: 1.3 false alarms per empty tile, 4x HERIDAL. Gate: fewer on NOMAD, same recall", False),
                          ("Find small people more often", "Gate: ≥ 85% of 8-16 px people, same file size", False)],
                         [("More varied aerial data (full WiSARD, SeaDronesSee)", "Gate: WiSARD recall > 61.3% without losing HERIDAL", False),
                          ("Measure on an ESP32-S3", "P4 checked 2026-11-09: same recall as the PC on 163 tiles", False)]])])
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
