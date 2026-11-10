"""HERIDAL split with a held-out-location validation set.

The original HERIDAL val list is a random 10% of the training photo sequences, so a val
photo usually has near-identical neighbours in train (train_BRA_1025 in val, 1024 and 1026
in train). Val recall then says little about the test photos, which come from separate
flights. This script builds val from whole location groups that never occur in test:

    VRD, GOR, JAS, RAK, BLA  (all their train and val photos)

Every other train/val photo goes to train. Test is unchanged (copy test.npz).

Val keeps every grid tile with a person and a fixed random sample of empty tiles, because
thresholds are picked on val recall only; val precision is therefore not comparable to
test precision.

    python scripts/prepare_heridal_groups.py <heridal_voc_root> <out_dir> [empty_val_tiles=4000]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_heridal import grid_tiles, save, split_names, train_tiles  # noqa: E402

VAL_GROUPS = ("VRD", "GOR", "JAS", "RAK", "BLA")


def group(name: str) -> str:
    return name.split("_")[1]


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    empty_keep = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    out.mkdir(parents=True, exist_ok=True)
    names = split_names(root, "train") + split_names(root, "val")
    val_names = sorted(n for n in names if group(n) in VAL_GROUPS)
    train_names = [n for n in names if group(n) not in VAL_GROUPS]
    test_groups = {group(n) for n in split_names(root, "test")}
    assert not test_groups & set(VAL_GROUPS), "val groups must not occur in test"
    print(f"train photos {len(train_names)}, val photos {len(val_names)} from {VAL_GROUPS}")
    (out / "split.txt").write_text("val\n" + "\n".join(val_names) + "\ntrain\n" + "\n".join(train_names) + "\n")

    rng = np.random.default_rng(0)
    save(out, "train", *train_tiles(root, train_names, rng))
    tiles, boxes = grid_tiles(root, val_names)
    person = [i for i, b in enumerate(boxes) if len(b)]
    empty = [i for i, b in enumerate(boxes) if not len(b)]
    keep = sorted(person + list(np.random.default_rng(1).choice(empty, min(empty_keep, len(empty)), replace=False)))
    print(f"val grid: {len(tiles)} tiles, {len(person)} with a person, keeping {len(keep)}")
    save(out, "val", [tiles[i] for i in keep], [boxes[i] for i in keep])


if __name__ == "__main__":
    main()
