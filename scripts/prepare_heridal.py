"""HERIDAL (Zenodo 5662351, CC BY 3.0) -> 160x160 training tiles with person boxes.

HERIDAL photos are 4000x3000 with people about 40-110 px tall. A drone camera
feeding a 160 px model sees the same person smaller or larger depending on
altitude, so tiles are cut at several crop sizes (320/480/640 px, i.e. 2-4x
downscale) to imitate different heights:

  * positive tiles: centred near a random person, jittered
  * random tiles: anywhere in the photo (mostly empty terrain = negatives)
  * CopyPaste tiles (PP-OCRv2's CopyPaste idea): a random tile with 1-3
    person patches from other photos pasted in at the same scale, so people
    appear on more kinds of background

Validation and test images are tiled on a regular grid at 4x downscale so
every annotated person is covered exactly once.

    python scripts/prepare_heridal.py D:/datasets/heridal/heridal_keras_retinanet_voc D:/datasets/voxhod
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 160
CROP_SIZES = (320, 480, 640)
POSITIVE_PER_IMAGE, RANDOM_PER_IMAGE, PASTE_PER_IMAGE = 8, 3, 2
GRID_CROP = 640


def read_boxes(xml_path: Path) -> np.ndarray:
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        box = obj.find("bndbox")
        boxes.append([float(box.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax")])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def cut(image: Image.Image, boxes: np.ndarray, x0: int, y0: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop [x0, x0+size) x [y0, y0+size), resize to TILE, keep boxes whose
    centre falls inside, clipped and rescaled to tile pixels."""
    tile = np.asarray(image.crop((x0, y0, x0 + size, y0 + size)).resize((TILE, TILE), Image.BILINEAR))
    if not len(boxes):
        return tile, np.zeros((0, 4), np.float32)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (centres[:, 0] >= x0) & (centres[:, 0] < x0 + size) & (centres[:, 1] >= y0) & (centres[:, 1] < y0 + size)
    kept = np.clip(boxes[inside] - [x0, y0, x0, y0], 0, size) * (TILE / size)
    return tile, kept.astype(np.float32)


def load(root: Path, name: str) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(root / "JPEGImages" / f"{name}.jpg").convert("RGB")
    xml = root / "Annotations" / f"{name}.xml"
    return image, read_boxes(xml) if xml.exists() else np.zeros((0, 4), np.float32)


def split_names(root: Path, split: str) -> list[str]:
    names = (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
    return [name for name in names if (root / "JPEGImages" / f"{name}.jpg").exists()]


def save(out: Path, split: str, tiles: list, boxes: list) -> None:
    lengths = np.array([len(b) for b in boxes], dtype=np.int32)
    flat = np.concatenate(boxes) if lengths.sum() else np.zeros((0, 4), np.float32)
    np.savez(out / f"{split}.npz", images=np.stack(tiles).astype(np.uint8), boxes=flat, counts=lengths)
    print(f"{split}: {len(tiles)} tiles, {int(lengths.sum())} persons, {int((lengths > 0).sum())} tiles with a person")


def train_tiles(root: Path, names: list[str], rng: np.random.Generator) -> tuple[list, list]:
    tiles, tile_boxes, patches = [], [], []
    for index, name in enumerate(names):
        image, boxes = load(root, name)
        width, height = image.size
        for box in boxes[:6]:  # person patch bank for CopyPaste, with a little context
            x0, y0, x1, y1 = box.astype(int)
            pad = max(4, int(0.15 * max(x1 - x0, y1 - y0)))
            patches.append(np.asarray(image.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))))
        for _ in range(POSITIVE_PER_IMAGE if len(boxes) else 0):
            size = int(rng.choice(CROP_SIZES))
            cx, cy = (boxes[rng.integers(len(boxes))].reshape(2, 2).mean(0) + rng.uniform(-0.4, 0.4, 2) * size)
            x0 = int(np.clip(cx - size / 2, 0, width - size)); y0 = int(np.clip(cy - size / 2, 0, height - size))
            tile, kept = cut(image, boxes, x0, y0, size)
            tiles.append(tile); tile_boxes.append(kept)
        for kind in ["random"] * RANDOM_PER_IMAGE + ["paste"] * PASTE_PER_IMAGE:
            size = int(rng.choice(CROP_SIZES))
            x0 = int(rng.integers(0, width - size)); y0 = int(rng.integers(0, height - size))
            tile, kept = cut(image, boxes, x0, y0, size)
            if kind == "paste" and patches:
                tile, kept = tile.copy(), list(kept)
                scale = TILE / size
                for _ in range(int(rng.integers(1, 4))):
                    patch = patches[int(rng.integers(len(patches)))]
                    ph, pw = max(2, int(patch.shape[0] * scale)), max(2, int(patch.shape[1] * scale))
                    if ph >= TILE or pw >= TILE:
                        continue
                    small = np.asarray(Image.fromarray(patch).resize((pw, ph), Image.BILINEAR))
                    if rng.random() < 0.5:
                        small = small[:, ::-1]
                    px, py = int(rng.integers(0, TILE - pw)), int(rng.integers(0, TILE - ph))
                    tile[py:py + ph, px:px + pw] = small
                    pad = 0.15 / 1.3 * np.array([pw, ph])  # the patch carries ~15% context on each side
                    kept.append([px + pad[0], py + pad[1], px + pw - pad[0], py + ph - pad[1]])
                kept = np.array(kept, np.float32).reshape(-1, 4)
            tiles.append(tile); tile_boxes.append(kept)
        if index % 200 == 0:
            print(f"  train {index}/{len(names)}", flush=True)
    return tiles, tile_boxes


def grid_tiles(root: Path, names: list[str]) -> tuple[list, list]:
    tiles, tile_boxes = [], []
    for name in names:
        image, boxes = load(root, name)
        width, height = image.size
        xs = list(range(0, width - GRID_CROP, GRID_CROP)) + [width - GRID_CROP]
        ys = list(range(0, height - GRID_CROP, GRID_CROP)) + [height - GRID_CROP]
        for y0 in ys:
            for x0 in xs:
                tile, kept = cut(image, boxes, x0, y0, GRID_CROP)
                tiles.append(tile); tile_boxes.append(kept)
    return tiles, tile_boxes


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    save(out, "train", *train_tiles(root, split_names(root, "train"), rng))
    save(out, "val", *grid_tiles(root, split_names(root, "val")))
    save(out, "test", *grid_tiles(root, split_names(root, "test")))


if __name__ == "__main__":
    main()
