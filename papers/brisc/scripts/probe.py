"""Estimate how much BRISC's train-test leakage inflates test scores, with frozen-embedding probes.

Removing leaked images from the test split leaves almost only no-tumor images, so the probes are
compared on the same full test split instead, under three training sets:

- ``full``: all of BRISC train;
- ``deleaked``: train without images of any Cheng patient that has a test image, and without
  train images whose thumbnail correlation with a test image reaches ``--corr``;
- ``random``: train without the same number of images per class, drawn at random, to separate
  the effect of leakage from the effect of a smaller training set.

A logistic regression and a 1-nearest-neighbour classifier are fit on DINOv2 embeddings. The
bootstrap resamples Cheng patients (single images where no patient is known) and reports macro
one-vs-rest AUROC and accuracy for each arm and the paired gaps, with 95% percentile intervals.
The random arm is repeated for several seeds; the interval is for seed 0 and the spread of the
point estimates across seeds is reported next to it.

    uv run python -I papers/brisc/scripts/probe.py --corr 0.99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from opendiagnostics.metrics.bootstrap import Estimate, bootstrap

FEATURES = Path("data/brisc/features")
OUT = Path("papers/brisc/generated")
SEEDS = 5


def deleaked_train(brisc: pd.DataFrame, patients: pd.DataFrame, corr: float) -> np.ndarray:
    pairs = pd.read_csv(FEATURES / "cross_pairs.csv", usecols=["train", "thumb_corr"])
    copies = set(pairs.loc[pairs["thumb_corr"] >= corr, "train"])
    sure = patients[patients["confident"]]
    test_patients = set(sure.loc[sure["split"] == "test", "patient_id"])
    shared = set(
        sure.loc[(sure["split"] == "train") & sure["patient_id"].isin(test_patients), "id"]
    )
    return ((brisc["split"] == "train") & ~brisc["id"].isin(copies | shared)).to_numpy()


def leaked_test(brisc: pd.DataFrame, patients: pd.DataFrame, corr: float) -> np.ndarray:
    pairs = pd.read_csv(FEATURES / "cross_pairs.csv", usecols=["test", "thumb_corr"])
    copies = set(pairs.loc[pairs["thumb_corr"] >= corr, "test"])
    sure = patients[patients["confident"]]
    train_patients = set(sure.loc[sure["split"] == "train", "patient_id"])
    shared = set(
        sure.loc[(sure["split"] == "test") & sure["patient_id"].isin(train_patients), "id"]
    )
    return ((brisc["split"] == "test") & brisc["id"].isin(copies | shared)).to_numpy()


def random_train(brisc: pd.DataFrame, keep: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(len(brisc), bool)
    train = (brisc["split"] == "train").to_numpy()
    for label in brisc["label"].unique():
        members = np.flatnonzero(train & (brisc["label"] == label).to_numpy())
        n = int((keep & (brisc["label"] == label).to_numpy()).sum())
        out[rng.choice(members, n, replace=False)] = True
    return out


def metrics(y: np.ndarray, prob: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    auroc = roc_auc_score(y[index], prob[index], multi_class="ovr", labels=range(prob.shape[1]))
    return float(auroc), float(accuracy_score(y[index], prob[index].argmax(1)))


def statistic(y: np.ndarray, probs: dict[str, np.ndarray]):
    def scores(index: np.ndarray) -> dict[str, float]:
        out = {}
        for arm, prob in probs.items():
            out[f"auroc_{arm}"], out[f"accuracy_{arm}"] = metrics(y, prob, index)
        for metric in ("auroc", "accuracy"):
            out[f"{metric}_full_minus_deleaked"] = out[f"{metric}_full"] - out[f"{metric}_deleaked"]
            out[f"{metric}_random_minus_deleaked"] = (
                out[f"{metric}_random"] - out[f"{metric}_deleaked"]
            )
        return out

    return scores


def rounded(estimate: Estimate) -> list[float]:
    return [round(estimate.value, 4), round(estimate.lower, 4), round(estimate.upper, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corr", type=float, required=True, help="thumbnail correlation cut")
    args = parser.parse_args()
    brisc = pd.read_csv(FEATURES / "brisc.csv")
    patients = pd.read_csv(OUT / "patients.csv.gz", dtype={"patient_id": str})
    embed = np.load(FEATURES / "brisc_embed.npy")
    classes = sorted(brisc["label"].unique())
    y = brisc["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    test = (brisc["split"] == "test").to_numpy()
    leaked = leaked_test(brisc, patients, args.corr)
    arms = {"full": (brisc["split"] == "train").to_numpy()}
    arms["deleaked"] = deleaked_train(brisc, patients, args.corr)
    randoms = [random_train(brisc, arms["deleaked"], seed) for seed in range(SEEDS)]
    groups = np.where(
        patients["confident"], "p" + patients["patient_id"].astype(str), patients["id"]
    )[test]
    summary: dict = {
        "corr": args.corr,
        "test_images": int(test.sum()),
        "train_images": {
            arm: {c: int((mask & (brisc["label"] == c).to_numpy()).sum()) for c in classes}
            for arm, mask in arms.items()
        },
        "leaked_test_images_by_class": {
            c: int((leaked & (brisc["label"] == c).to_numpy()).sum()) for c in classes
        },
        "bootstrap_unit": "Cheng patient where matched, otherwise image; 1000 replicates",
        "probes": {},
    }
    probes = {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "knn1": lambda: KNeighborsClassifier(n_neighbors=1, metric="cosine"),
    }
    for name, make in probes.items():

        def predict(mask: np.ndarray, make=make) -> np.ndarray:
            return make().fit(embed[mask], y[mask]).predict_proba(embed[test])

        probs = {arm: predict(mask) for arm, mask in arms.items()}
        random_probs = [predict(mask) for mask in randoms]
        probs["random"] = random_probs[0]
        result = bootstrap(statistic(y[test], probs), int(test.sum()), list(groups))
        every = np.arange(int(test.sum()))
        spread = np.array([metrics(y[test], p, every) for p in random_probs])
        summary["probes"][name] = {
            **{k: rounded(v) for k, v in result.items()},
            "random_seeds_auroc_min_max": [
                round(float(spread[:, 0].min()), 4),
                round(float(spread[:, 0].max()), 4),
            ],
            "random_seeds_accuracy_min_max": [
                round(float(spread[:, 1].min()), 4),
                round(float(spread[:, 1].max()), 4),
            ],
        }
    text = json.dumps(summary, indent=2)
    (OUT / f"probe_corr{args.corr}.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
