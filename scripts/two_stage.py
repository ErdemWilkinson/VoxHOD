"""Two-stage detector from cached heatmaps: nano flags tiles, a second model re-checks only those.

Per pair (stage-1 run, output name): the gate tau is chosen on val only, as the highest threshold at which
the tiles whose stage-1 maximum is >= tau still contain >= COVER of the val people. The combined heatmap is
the stage-2 heatmap on passed tiles and 0 elsewhere, written as <cache>/<out>_<split>.npy so that
arms_hier.py and topk_recall.py can score it like any other run.

    python scripts/two_stage.py <cache_dir> <data_dir> <stage2> K1L=C1 K2L=C2 ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import THRESHOLDS  # noqa: E402

COVER = float(os.environ.get("TWO_STAGE_COVER", "0.97"))
SPLITS = ("val", "test", "wisard", "nomad_tn_test")


def main() -> None:
    cache, data, stage2 = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    pairs = [a.split("=", 1) for a in sys.argv[4:]]
    people = np.load(data / "val.npz")["counts"].astype(float)
    second = {s: np.load(cache / f"{stage2}_{s}.npy") for s in SPLITS}
    report = {"cover": COVER, "stage2": stage2, "pairs": {}}
    for first, out in pairs:
        tile_max = {s: np.load(cache / f"{first}_{s}.npy").reshape(-1, 1600).max(1).astype(np.float32) for s in SPLITS}
        ok = [t for t in THRESHOLDS if people[tile_max["val"] >= t].sum() / people.sum() >= COVER]
        tau = float(max(ok))
        r = {"stage1": first, "tau": tau}
        for s in SPLITS:
            keep = tile_max[s] >= tau
            combined = np.where(keep[:, None, None], second[s], 0).astype(np.float16)
            np.save(cache / f"{out}_{s}.npy", combined)
            r[f"passed_{s}"] = float(keep.mean())
        r["val_people_covered"] = float(people[tile_max["val"] >= tau].sum() / people.sum())
        report["pairs"][out] = r
        print(out, json.dumps(r))
    (cache / f"two_stage_{stage2}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
