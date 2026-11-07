"""Draw the VoxHOD README charts into docs/ (light and dark SVG).

Numbers at the fixed threshold 0.2, int8 files (v3 QAT, nano PTQ):
artifacts_v3/evaluation.json, artifacts_v3/evaluation_wisard.json,
artifacts_nano/evaluation.json. The diagnosis numbers (v3, HERIDAL test set,
threshold 0.2, 3x3 peaks) come from a separate analysis run on 2026-11-07.

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
    roadmap(DOCS / "roadmap-voxhod", "VoxHOD roadmap",
            "Rule: the device model never grows; larger models are only teachers. A step ships only if it passes its gate.",
            ["Next", "This week", "Later"],
            [("VoxHOD", [[("Train with many more empty and hard-negative tiles", "Gate: precision > 20.4% at recall ≥ 83% and WiSARD recall ≥ 61.3%", True)],
                         [("Find small people more often", "Gate: ≥ 85% of 8-16 px people, same file size", False)],
                         [("More varied aerial data (full WiSARD, SeaDronesSee)", "Gate: WiSARD recall > 61.3% without losing HERIDAL", False),
                          ("Check device output against the PC, measure on an ESP32-S3", "Gate: same detections on device and PC", False)]])])
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
