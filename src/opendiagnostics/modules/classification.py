"""Binary, multiclass, and multilabel image classification."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.metrics import classification as cm
from opendiagnostics.modules.base import DiagnosticModule, Prediction


class ImageClassifier(DiagnosticModule):
    """Any network that maps images to logits, e.g. ``timm.create_model`` or a MONAI net.

    Binary tasks need one output logit (BCE); multiclass needs one per class
    (cross-entropy); multilabel needs one per label (BCE). ``threshold`` is the binary
    decision threshold, ``youden`` to fit it on validation. ``tta`` averages the
    probabilities of the image and its horizontal flip at evaluation.
    """

    default_monitor = "auroc"

    def __init__(
        self,
        network: nn.Module,
        info: DatasetInfo | None = None,
        threshold: float | Literal["youden"] = "youden",
        label_smoothing: float = 0.0,
        pos_weight: list[float] | None = None,
        class_weight: list[float] | None = None,
        tta: bool = False,
        calibration_bins: int = 15,
    ) -> None:
        super().__init__(network, info)
        info = self.info
        if info.label_mode in ("detection", "segmentation"):
            raise ValueError("ImageClassifier needs a classification data module")
        self.mode = info.label_mode
        self.outputs = 1 if self.mode == "binary" else info.num_classes
        self.threshold = threshold
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        if class_weight is not None and (
            self.mode != "multiclass" or len(class_weight) != info.num_classes
        ):
            raise ValueError(
                f"class_weight needs one weight per class of a multiclass task; got "
                f"{len(class_weight)} for a {self.mode} task with {info.num_classes} classes. "
                "Use pos_weight for binary and multilabel tasks."
            )
        self.class_weight = class_weight
        self.tta = tta
        self.calibration_bins = calibration_bins

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(images)
        if logits.ndim != 2 or logits.shape[1] != self.outputs:
            raise ValueError(
                f"the network returned logits shaped {tuple(logits.shape)}; a {self.mode} task "
                f"with {self.info.num_classes} classes needs (batch, {self.outputs}). Set the "
                "network's num_classes (timm) or out_channels (MONAI) to that."
            )
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits, target = self(batch["image"]), batch["target"]
        if self.mode == "multiclass":
            weight = None if self.class_weight is None else logits.new_tensor(self.class_weight)
            loss = nn.functional.cross_entropy(
                logits, target, weight=weight, label_smoothing=self.label_smoothing
            )
        else:
            weight = None if self.pos_weight is None else logits.new_tensor(self.pos_weight)
            smoothed = target.float() * (1 - self.label_smoothing) + self.label_smoothing / 2
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.reshape(target.shape[0], -1),
                smoothed.reshape(target.shape[0], -1),
                pos_weight=weight,
            )
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=target.shape[0])
        return loss

    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Class probabilities, averaged over the flipped view when ``tta`` is on."""
        views = [images, images.flip(-1)] if self.tta else [images]
        probs = []
        for view in views:
            logits = self(view).float()
            probs.append(logits.softmax(1) if self.mode == "multiclass" else logits.sigmoid())
        return torch.stack(probs).mean(0)

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> list[Prediction]:
        probs = self.probabilities(batch["image"]).cpu().numpy()
        target = batch["target"].cpu().numpy()
        return [
            {"id": i, "patient_id": p, "target": target[n], "prob": probs[n]}
            for n, (i, p) in enumerate(zip(batch["id"], batch["patient_id"], strict=True))
        ]

    @staticmethod
    def _arrays(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack([np.asarray(p["target"]) for p in predictions]),
            np.stack([np.asarray(p["prob"]) for p in predictions]),
        )

    def calibrate(self, predictions: list[Prediction]) -> dict[str, Any]:
        if self.mode != "binary":
            return {}
        if self.threshold != "youden":
            return {"threshold": float(self.threshold), "threshold_source": "config"}
        target, prob = self._arrays(predictions)
        return {
            "threshold": cm.youden_threshold(target, prob[:, 0]),
            "threshold_source": "youden on val",
        }

    def score(
        self,
        predictions: list[Prediction],
        calibration: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target, prob = self._arrays(predictions)
        names = self.info.class_names
        threshold = (calibration or {}).get("threshold", 0.5)

        def statistic(index: np.ndarray) -> dict[str, float]:
            t, p = target[index], prob[index]
            if self.mode == "binary":
                return {
                    "accuracy_at_0.5": cm.threshold_metrics(t, p[:, 0], 0.5)["accuracy"],
                    "auroc": cm.auroc(t, p[:, 0]),
                    "auprc": cm.auprc(t, p[:, 0]),
                    "brier": cm.brier(t, p[:, 0]),
                    "ece": cm.expected_calibration_error(t, p[:, 0], self.calibration_bins),
                } | cm.threshold_metrics(t, p[:, 0], threshold)
            roc = cm.per_class(t, p, cm.auroc)
            prc = cm.per_class(t, p, cm.auprc)
            out = {"auroc": cm.nanmean(roc), "auprc": cm.nanmean(prc)}
            if self.mode == "multiclass":
                predicted = p.argmax(1)
                out["accuracy"] = float(np.mean(predicted == t))
                out["balanced_accuracy"] = cm.balanced_accuracy(t, predicted, len(names))
            else:
                out["accuracy"] = float(np.mean((p >= 0.5) == t))
            for name, a, b in zip(names, roc, prc, strict=True):
                out[f"auroc/{name}"] = float(a)
                out[f"auprc/{name}"] = float(b)
            return out

        patients = [p["patient_id"] for p in predictions]
        result = self.bootstrapped(statistic, len(predictions), patients, bootstrap)
        result["n_samples"] = len(predictions)
        result["n_patients"] = len(set(patients))
        if self.mode == "binary":
            result["prevalence"] = float(np.mean(target))
            result["threshold"] = threshold
        return result

    def prediction_rows(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        names = self.info.class_names
        rows = []
        for p in predictions:
            row: dict[str, Any] = {"id": p["id"], "patient_id": p["patient_id"]}
            target = np.asarray(p["target"]).reshape(-1)
            if self.mode == "multilabel":
                row |= {f"target/{n}": float(v) for n, v in zip(names, target, strict=True)}
            else:
                row["target"] = int(target[0])
            if self.mode == "binary":
                row[f"prob/{names[1]}"] = float(p["prob"][0])
            else:
                row |= {f"prob/{n}": float(v) for n, v in zip(names, p["prob"], strict=True)}
            rows.append(row)
        return rows
