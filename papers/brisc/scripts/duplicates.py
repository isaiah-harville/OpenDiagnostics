"""Count exact, near and semantic duplicate pairs within and across BRISC's train and test splits.

Reads the features that ``features.py`` wrote and compares every BRISC image with every other:

- exact: the SHA-256 of the decoded grayscale pixels (and of the file bytes);
- perceptual: the Hamming distance between 64-bit pHashes, taking the smaller of the plain and
  the left-right flipped hash;
- thumbnail: the correlation of 32 x 32 normalized thumbnails, the best over flips, a 180 degree
  turn and 90 and 80 percent centre crops of either image;
- semantic: the cosine similarity of DINOv2 ViT-S/14 embeddings.

Writes ``papers/brisc/generated/duplicates.json`` (counts per level, threshold, split pair and
class) and ``data/brisc/features/cross_pairs.csv`` (every cross-split pair that passes the
loosest threshold of any level, with all three scores, for the contact sheets):

    uv run python -I papers/brisc/scripts/duplicates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURES = Path("data/brisc/features")
OUT = Path("papers/brisc/generated")
THRESHOLDS = {
    "thumb_corr": (0.99, 0.97, 0.95, 0.90),
    "phash_hamming": (0, 4, 8, 12),
    "embed_cosine": (0.98, 0.95, 0.90, 0.85),
}
LOOSE = {"thumb_corr": 0.90, "phash_hamming": 12, "embed_cosine": 0.85}


def thumb_similarity(thumbs: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(thumbs)
    base = t[:, 0]
    best = base @ base.T
    for v in range(1, t.shape[1]):
        s = base @ t[:, v].T
        best = torch.maximum(best, torch.maximum(s, s.T))
    return best.numpy()


def bits(hashes: pd.Series) -> np.ndarray:
    values = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    shifts = np.arange(64, dtype=np.uint64)
    return ((values[:, None] >> shifts) & np.uint64(1)).astype(np.float32)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rint(a @ (1 - b).T + (1 - a) @ b.T).astype(np.int16)


def phash_distance(table: pd.DataFrame) -> np.ndarray:
    plain, flipped = bits(table["phash"]), bits(table["phash_hflip"])
    return np.minimum(hamming(plain, plain), hamming(plain, flipped))


def passes(level: str, score: np.ndarray, threshold: float) -> np.ndarray:
    return score <= threshold if level == "phash_hamming" else score >= threshold


def count(table: pd.DataFrame, hit: np.ndarray) -> dict:
    """Pairs per split pair, and test images with a train partner, overall and per class."""
    split = table["split"].to_numpy()
    label = table["label"].to_numpy()
    i, j = np.nonzero(np.triu(hit, 1))
    kinds = pd.Series(
        ["-".join(sorted((split[a], split[b]), reverse=True)) for a, b in zip(i, j, strict=True)],
        dtype=object,
    )
    cross = (split[i] == "train") ^ (split[j] == "train")
    train, test = split == "train", split == "test"
    test_hit = hit[np.ix_(test, train)].any(1)
    test_labels = label[test]
    return {
        "pairs": {k: int((kinds == k).sum()) for k in ("train-train", "train-test", "test-test")},
        "cross_pairs_label_disagrees": int((label[i] != label[j])[cross].sum()),
        "test_images_with_train_partner": int(test_hit.sum()),
        "test_images_with_train_partner_by_class": {
            c: int(test_hit[test_labels == c].sum()) for c in sorted(set(label))
        },
        "train_images_with_test_partner": int(hit[np.ix_(train, test)].any(1).sum()),
        "images_in_any_pair": int(hit.any(1).sum()),
    }


def exact(table: pd.DataFrame, column: str) -> dict:
    groups = table.groupby(column)
    sizes = groups.size()
    dup = sizes[sizes > 1]
    hit = table[column].to_numpy()[:, None] == table[column].to_numpy()[None, :]
    np.fill_diagonal(hit, False)
    i, j = np.nonzero(np.triu(hit, 1))
    plane = table["plane"].to_numpy()
    return {
        "duplicate_groups": len(dup),
        "images_in_groups": int(dup.sum()),
        "pairs_with_different_plane_tags": int((plane[i] != plane[j]).sum()),
        **count(table, hit),
    }


def tally(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame["split"] + "/" + frame[column]).value_counts().sort_index()


def main() -> None:
    table = pd.read_csv(FEATURES / "brisc.csv")
    scores = {
        "thumb_corr": thumb_similarity(np.load(FEATURES / "brisc_thumbs.npy")),
        "phash_hamming": phash_distance(table),
        "embed_cosine": (lambda e: e @ e.T)(np.load(FEATURES / "brisc_embed.npy")),
    }
    n = len(table)
    diagonal = np.eye(n, dtype=bool)
    summary: dict = {
        "images": {str(k): int(n) for k, n in tally(table, "label").items()},
        "images_by_plane": {str(k): int(n) for k, n in tally(table, "plane").items()},
        "label_matches_folder": bool((table["label"] == table["folder"]).all()),
        "exact_pixels": exact(table, "pixel_sha256"),
        "exact_file_bytes": exact(table, "file_sha256"),
    }
    for level, thresholds in THRESHOLDS.items():
        score = scores[level]
        summary[level] = {
            str(t): count(table, passes(level, score, t) & ~diagonal) for t in thresholds
        }

    split = table["split"].to_numpy()
    cross = (split[:, None] == "train") & (split[None, :] == "test")
    loose = np.zeros_like(cross)
    for level, threshold in LOOSE.items():
        loose |= passes(level, scores[level], threshold)
    i, j = np.nonzero(cross & loose)
    pairs = pd.DataFrame(
        {
            "train": table["id"].to_numpy()[i],
            "test": table["id"].to_numpy()[j],
            "train_label": table["label"].to_numpy()[i],
            "test_label": table["label"].to_numpy()[j],
            "train_plane": table["plane"].to_numpy()[i],
            "test_plane": table["plane"].to_numpy()[j],
            **{level: scores[level][i, j] for level in THRESHOLDS},
        }
    )
    pairs.to_csv(FEATURES / "cross_pairs.csv", index=False)
    summary["cross_pairs_at_loosest_thresholds"] = len(pairs)
    text = json.dumps(summary, indent=2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "duplicates.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
