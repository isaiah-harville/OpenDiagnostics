"""Recover Cheng et al. patient IDs for BRISC images and count patients that span train and test.

Each BRISC image is matched to the Cheng figshare slice whose canonical thumbnail correlates best
with it, the best over the same flips and crops ``duplicates.py`` uses. A match is confident when
the correlation clears a threshold and beats the best slice of any *other* patient by a margin
(the runner-up from the same patient is usually an adjacent slice and says nothing about which
patient it is). BRISC's no-tumor images cannot come from Cheng's set, which has no healthy
scans, so their best correlations give the null distribution the threshold must clear. The best
DINOv2 neighbour is reported as an independent second opinion on the patient.

    uv run python -I papers/brisc/scripts/patients.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURES = Path("data/brisc/features")
OUT = Path("papers/brisc/generated")
THRESHOLD = 0.95
MARGIN = 0.02


def similarity(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q, b = torch.from_numpy(queries), torch.from_numpy(bank)
    best = q[:, 0] @ b[:, 0].T
    for v in range(1, q.shape[1]):
        best = torch.maximum(best, q[:, 0] @ b[:, v].T)
        best = torch.maximum(best, q[:, v] @ b[:, 0].T)
    return best.numpy()


def match(score: np.ndarray, patients: np.ndarray) -> pd.DataFrame:
    best = score.argmax(1)
    first = score[np.arange(len(score)), best]
    other = np.where(patients[None, :] == patients[best][:, None], -np.inf, score)
    return pd.DataFrame(
        {"cheng_index": best, "correlation": first, "other_patient_best": other.max(1)}
    )


def tally(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame["split"] + "/" + frame[column]).value_counts().sort_index()


def main() -> None:
    brisc = pd.read_csv(FEATURES / "brisc.csv")
    cheng = pd.read_csv(FEATURES / "cheng.csv", dtype={"patient_id": str})
    patients = cheng["patient_id"].to_numpy()
    score = similarity(
        np.load(FEATURES / "brisc_thumbs.npy"), np.load(FEATURES / "cheng_thumbs.npy")
    )
    found = match(score, patients)
    embed = np.load(FEATURES / "brisc_embed.npy") @ np.load(FEATURES / "cheng_embed.npy").T
    embed_best = embed.argmax(1)

    table = brisc[["id", "split", "label", "plane"]].copy()
    table["cheng_id"] = cheng["id"].to_numpy()[found["cheng_index"]]
    table["cheng_label"] = cheng["label"].to_numpy()[found["cheng_index"]]
    table["patient_id"] = patients[found["cheng_index"]]
    table["correlation"] = found["correlation"].round(4)
    table["other_patient_best"] = found["other_patient_best"].round(4)
    table["embed_cosine"] = embed[np.arange(len(embed)), embed_best].round(4)
    table["embed_same_patient"] = patients[embed_best] == table["patient_id"].to_numpy()
    null = table.loc[table["label"] == "no_tumor", "correlation"]
    table["confident"] = (
        (table["label"] != "no_tumor")
        & (table["correlation"] >= THRESHOLD)
        & (table["correlation"] - table["other_patient_best"] >= MARGIN)
    )
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "patients.csv.gz", index=False)

    sure = table[table["confident"]]
    by_split = {s: set(sure.loc[sure["split"] == s, "patient_id"]) for s in ("train", "test")}
    shared = by_split["train"] & by_split["test"]
    test_sure = sure[sure["split"] == "test"]
    leaked = test_sure["patient_id"].isin(shared)
    reuse = sure.groupby("cheng_id")["split"].agg(list)
    summary = {
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "tumor_images": {
            s: int(((table["split"] == s) & (table["label"] != "no_tumor")).sum())
            for s in ("train", "test")
        },
        "no_tumor_null_correlation": {
            "max": round(float(null.max()), 4),
            "q99": round(float(null.quantile(0.99)), 4),
            "median": round(float(null.median()), 4),
            "at_or_above_threshold": int((null >= THRESHOLD).sum()),
        },
        "tumor_correlation_quantiles": {
            q: round(float(table.loc[table["label"] != "no_tumor", "correlation"].quantile(q)), 4)
            for q in (0.05, 0.25, 0.5)
        },
        "confident": {s: int((sure["split"] == s).sum()) for s in ("train", "test")},
        "confident_by_class": {str(k): int(n) for k, n in tally(sure, "label").items()},
        "confident_label_agrees_with_cheng": int((sure["label"] == sure["cheng_label"]).sum()),
        "confident_embed_same_patient": int(sure["embed_same_patient"].sum()),
        "cheng_patients": int(cheng["patient_id"].nunique()),
        "patients": {s: len(p) for s, p in by_split.items()},
        "patients_in_train_and_test": len(shared),
        "test_images_from_shared_patients": int(leaked.sum()),
        "test_images_from_shared_patients_by_class": {
            c: int(n) for c, n in test_sure[leaked].groupby("label").size().items()
        },
        "test_images_whose_cheng_slice_is_in_train": int(
            test_sure["cheng_id"].isin(set(sure.loc[sure["split"] == "train", "cheng_id"])).sum()
        ),
        "confident_margin_quantiles": {
            q: round(float((sure["correlation"] - sure["other_patient_best"]).quantile(q)), 4)
            for q in (0.01, 0.1, 0.5)
        },
        "confident_label_disagreements": sure.loc[
            sure["label"] != sure["cheng_label"], ["id", "label", "cheng_id", "cheng_label"]
        ].to_dict("records"),
        "cheng_slices_matched": int(sure["cheng_id"].nunique()),
        "cheng_slices_matched_more_than_once": int((reuse.str.len() > 1).sum()),
        "cheng_slices_matched_in_both_splits": int(
            reuse.apply(lambda s: "train" in s and "test" in s).sum()
        ),
    }
    text = json.dumps(summary, indent=2)
    (OUT / "patients.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
