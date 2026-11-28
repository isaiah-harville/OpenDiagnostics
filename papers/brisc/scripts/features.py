"""Compute per-image features for the BRISC duplicate and patient-leakage audit.

For every BRISC classification image and every slice of Cheng et al.'s figshare set this writes
a pixel hash, a perceptual hash, normalized thumbnails under several geometric variants, and a
DINOv2 embedding. Every image is first put into a canonical frame: grayscale, cropped to the
bounding box of its foreground, and padded to a square, so that border trimming and resizing
do not change its thumbnail. Cheng slices are min-max scaled to 8 bits, as Cheng's README does,
and saved as PNG in ``data/brisc/cheng/png`` for the embedder and the contact sheets.

Run from the repository root after unpacking the Mendeley zip into ``data/brisc/brisc`` and the
four figshare zips into ``data/brisc/cheng/mat``:

    uv run --with imagehash python -I papers/brisc/scripts/features.py
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import imagehash  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
import timm
import timm.data
import torch
from PIL import Image

DATA = Path("data/brisc")
THUMB = 32
CROPS = (0.9, 0.8)
VARIANTS = ("identity", "hflip", "vflip", "rot180", "crop90", "crop80")
CLASSES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor"}
CHENG_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
EMBEDDER = "vit_small_patch14_dinov2.lvd142m"


def canonical(gray: np.ndarray) -> Image.Image:
    """Crop to the foreground bounding box and pad to a centred square."""
    mask = gray > 0.1 * float(gray.max())
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) and len(cols):
        gray = gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    side = max(gray.shape)
    square = np.zeros((side, side), np.uint8)
    top, left = (side - gray.shape[0]) // 2, (side - gray.shape[1]) // 2
    square[top : top + gray.shape[0], left : left + gray.shape[1]] = gray
    return Image.fromarray(square)


def unit(image: Image.Image) -> np.ndarray:
    flat = np.asarray(image.resize((THUMB, THUMB), Image.Resampling.BOX), np.float32).ravel()
    flat = flat - flat.mean()
    return flat / max(float(np.linalg.norm(flat)), 1e-6)


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    side = image.size[0]
    inset = round(side * (1 - fraction) / 2)
    return image.crop((inset, inset, side - inset, side - inset))


def thumbnails(image: Image.Image) -> np.ndarray:
    flipped = (
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        image.transpose(Image.Transpose.ROTATE_180),
    )
    return np.stack([unit(v) for v in flipped] + [unit(center_crop(image, f)) for f in CROPS])


def describe(gray: np.ndarray) -> dict:
    image = canonical(gray)
    return {
        "pixel_sha256": hashlib.sha256(
            np.asarray(gray.shape, np.int64).tobytes() + gray.tobytes()
        ).hexdigest(),
        "phash": str(imagehash.phash(image)),
        "phash_hflip": str(imagehash.phash(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))),
        "thumbs": thumbnails(image),
    }


def brisc_item(path: Path) -> dict:
    with Image.open(path) as image:
        size, mode = image.size, image.mode
        gray = np.asarray(image.convert("L"))
    _, split, number, code, plane, _ = path.stem.split("_")
    return {
        "id": path.stem,
        "split": split,
        "number": int(number),
        "label": CLASSES[code],
        "folder": path.parent.name,
        "plane": plane,
        "width": size[0],
        "height": size[1],
        "mode": mode,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
        **describe(gray),
    }


def read_cheng(path: Path) -> tuple[np.ndarray, str, int]:
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["cjdata/image"], np.float64).T
        pid = "".join(chr(int(c)) for c in np.asarray(f["cjdata/PID"]).ravel())
        label = int(np.asarray(f["cjdata/label"]).ravel()[0])
    span = max(raw.max() - raw.min(), 1.0)
    return np.round(255 * (raw - raw.min()) / span).astype(np.uint8), pid, label


def cheng_item(path: Path) -> dict:
    gray, pid, label = read_cheng(path)
    png = path.parent.parent / "png" / f"{path.stem}.png"
    Image.fromarray(gray).save(png)
    return {
        "id": f"cheng_{int(path.stem):04d}",
        "split": "cheng",
        "number": int(path.stem),
        "label": CHENG_LABELS[label],
        "patient_id": pid,
        "width": gray.shape[1],
        "height": gray.shape[0],
        "path": str(png),
        **describe(gray),
    }


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def embed(paths: list[Path], batch: int = 32) -> np.ndarray:
    model = timm.create_model(EMBEDDER, pretrained=True, num_classes=0).eval()
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**config)
    out = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch):
            images = [
                transform(canonical(load_gray(p)).convert("RGB"))
                for p in paths[start : start + batch]
            ]
            features = model(torch.stack(images))
            out.append(torch.nn.functional.normalize(features, dim=1).numpy())
            print(f"embedded {start + len(images)}/{len(paths)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DATA / "features")
    args = parser.parse_args()
    torch.set_num_threads(args.workers)
    brisc_paths = sorted((DATA / "brisc/brisc2025/classification_task").glob("*/*/*.jpg"))
    cheng_paths = sorted((DATA / "cheng/mat").glob("*.mat"), key=lambda p: int(p.stem))
    (DATA / "cheng/png").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        brisc = list(pool.map(brisc_item, brisc_paths, chunksize=64))
        cheng = list(pool.map(cheng_item, cheng_paths, chunksize=64))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in (("brisc", brisc), ("cheng", cheng)):
        table = pd.DataFrame([{k: v for k, v in i.items() if k != "thumbs"} for i in items])
        table.to_csv(args.out / f"{name}.csv", index=False)
        np.save(args.out / f"{name}_thumbs.npy", np.stack([i["thumbs"] for i in items]))
        np.save(args.out / f"{name}_embed.npy", embed([Path(p) for p in table["path"]]))
        print(f"{name}: {len(table)} images")


if __name__ == "__main__":
    main()
