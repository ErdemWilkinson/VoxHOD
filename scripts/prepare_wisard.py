"""WiSARD multi-modal sample (visual frames, MIT-worded licence) -> an external test set.

One clip from a different country, camera and terrain than HERIDAL, so it shows
how the model generalizes. It has 264 correlated frames and is never trained on.
Tiles are 640 px grid crops resized to 160, the same scale as the HERIDAL test.

    python scripts/prepare_wisard.py D:/datasets/raw/wisard/sample.zip D:/datasets/voxhod
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_heridal import GRID_CROP, cut, save


def main() -> None:
    source, out = sys.argv[1], Path(sys.argv[2])
    tiles, tile_boxes = [], []
    with zipfile.ZipFile(source) as archive:
        for name in sorted(n for n in archive.namelist() if "_VIS_" in n and n.endswith(".jpeg")):
            image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
            width, height = image.size
            label = archive.read(name[:-5] + ".txt").decode().split("\n")
            rows = np.array([line.split()[1:] for line in label if line.strip()], np.float32).reshape(-1, 4)
            boxes = np.stack([(rows[:, 0] - rows[:, 2] / 2) * width, (rows[:, 1] - rows[:, 3] / 2) * height,
                              (rows[:, 0] + rows[:, 2] / 2) * width, (rows[:, 1] + rows[:, 3] / 2) * height], 1)
            size = min(GRID_CROP, width, height)
            xs = list(range(0, width - size, size)) + [width - size]
            ys = list(range(0, height - size, size)) + [height - size]
            for y0 in ys:
                for x0 in xs:
                    tile, kept = cut(image, boxes, x0, y0, size)
                    tiles.append(tile); tile_boxes.append(kept)
    print("image size", width, height, "crop", size)
    save(out, "wisard", tiles, tile_boxes)


if __name__ == "__main__":
    main()
