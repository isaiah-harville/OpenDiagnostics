"""Draw contact sheets of cross-split BRISC pairs and shared Cheng patients for review by eye.

For each similarity level and score band, one PNG shows the strongest pairs in the band and a
random sample from it, each as train image | test image with both IDs and all three scores.
``patients_*.png`` shows, for test images whose Cheng patient also appears in train, the test
image, its matched Cheng slice, and the train image of that patient closest in Cheng slice order.

Sheets go to ``papers/brisc/generated/sheets/``, which git ignores: they hold dataset images.

    uv run python -I papers/brisc/scripts/sheets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

FEATURES = Path("data/brisc/features")
OUT = Path("papers/brisc/generated/sheets")
TILE = 192
PER_SHEET = 16
BANDS = {
    "thumb_corr": ((0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.90, 0.95)),
    "phash_hamming": ((0, 4), (5, 8), (9, 12)),
    "embed_cosine": ((0.98, 1.01), (0.95, 0.98), (0.90, 0.95), (0.85, 0.90)),
}


def tile(path: str, caption: str) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("L")
        image.thumbnail((TILE, TILE))
    canvas = Image.new("L", (TILE, TILE + 28), 255)
    canvas.paste(image, ((TILE - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    for k, line in enumerate(caption.split("\n")[:2]):
        draw.text((2, TILE + 2 + 13 * k), line, fill=0)
    return canvas


def sheet(rows: list[list[Image.Image]], path: Path, columns: int = 4) -> None:
    width = len(rows[0]) * TILE + 12
    height = TILE + 28 + 8
    grid = Image.new("L", (columns * width, -(-len(rows) // columns) * height), 200)
    for k, row in enumerate(rows):
        x, y = (k % columns) * width, (k // columns) * height
        for m, image in enumerate(row):
            grid.paste(image, (x + m * TILE, y))
    grid.save(path)


def short(name: str) -> str:
    return name.replace("brisc2025_", "").replace("_t1", "")


def pair_sheets(pairs: pd.DataFrame, paths: dict[str, str]) -> None:
    rng = np.random.default_rng(0)
    for level, bands in BANDS.items():
        ascending = level == "phash_hamming"
        for low, high in bands:
            band = pairs[(pairs[level] >= low) & (pairs[level] < high + (1 if ascending else 0))]
            if band.empty:
                continue
            ordered = band.sort_values(level, ascending=ascending)
            top = ordered.head(PER_SHEET // 2)
            rest = ordered.iloc[len(top) :]
            sample = rest.iloc[
                np.sort(rng.choice(len(rest), min(len(rest), PER_SHEET // 2), replace=False))
            ]
            rows = []
            for _, r in pd.concat([top, sample]).iterrows():
                scores = f"c{r.thumb_corr:.3f} h{r.phash_hamming} e{r.embed_cosine:.3f}"
                rows.append(
                    [
                        tile(paths[r.train], f"{short(r.train)}\n{scores}"),
                        tile(paths[r.test], f"{short(r.test)}\n{r.test_label}"),
                    ]
                )
            sheet(rows, OUT / f"{level}_{low}-{high}_n{len(band)}.png")


def patient_sheets(patients: pd.DataFrame, cheng: pd.DataFrame, paths: dict[str, str]) -> None:
    sure = patients[patients["confident"]]
    shared = set(sure.loc[sure["split"] == "train", "patient_id"]) & set(
        sure.loc[sure["split"] == "test", "patient_id"]
    )
    number = dict(zip(cheng["id"], cheng["number"], strict=True))
    cheng_paths = dict(zip(cheng["id"], cheng["path"], strict=True))
    rows = []
    for _, r in sure[(sure["split"] == "test") & sure["patient_id"].isin(shared)].iterrows():
        train = sure[(sure["split"] == "train") & (sure["patient_id"] == r.patient_id)]
        gap = (train["cheng_id"].map(number) - number[r.cheng_id]).abs()
        nearest = train.loc[gap.idxmin()]
        rows.append(
            [
                tile(paths[r.id], f"{short(r.id)}\nc{r.correlation:.3f} p{r.patient_id}"),
                tile(cheng_paths[r.cheng_id], f"{r.cheng_id}\n{r.cheng_label}"),
                tile(paths[nearest.id], f"{short(nearest.id)}\ngap {int(gap.min())} slices"),
            ]
        )
    for start in range(0, len(rows), PER_SHEET):
        sheet(rows[start : start + PER_SHEET], OUT / f"patients_{start // PER_SHEET:02d}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    brisc = pd.read_csv(FEATURES / "brisc.csv")
    paths = dict(zip(brisc["id"], brisc["path"], strict=True))
    pair_sheets(pd.read_csv(FEATURES / "cross_pairs.csv"), paths)
    patients = pd.read_csv(
        Path("papers/brisc/generated/patients.csv.gz"), dtype={"patient_id": str}
    )
    patient_sheets(patients, pd.read_csv(FEATURES / "cheng.csv", dtype={"patient_id": str}), paths)
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
