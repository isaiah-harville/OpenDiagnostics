import numpy as np
import pytest
import torch
from torch import nn

from opendiagnostics.data import DatasetInfo
from opendiagnostics.modules import ImageClassifier


def _classifier(mode, names, **kwargs):
    info = DatasetInfo(label_mode=mode, class_names=tuple(names), in_channels=1)
    outputs = 1 if mode == "binary" else len(names)
    network = nn.Sequential(nn.Flatten(), nn.Linear(16, outputs))
    return ImageClassifier(network, info, **kwargs)


def _batch(target, n=6):
    return {
        "image": torch.rand(n, 1, 4, 4, generator=torch.Generator().manual_seed(0)),
        "target": target,
        "id": [f"s{i}" for i in range(n)],
        "patient_id": [f"p{i}" for i in range(n)],
    }


def test_multilabel_scores_each_label_and_writes_one_column_per_label():
    model = _classifier("multilabel", ["effusion", "mass"])
    target = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    predictions = model.predict_batch(_batch(target))
    scores = model.score(predictions)
    assert {"auroc", "auprc", "accuracy", "auroc/effusion", "auroc/mass"} <= set(scores)
    probs = np.stack([p["prob"] for p in predictions])
    assert scores["accuracy"] == pytest.approx(np.mean((probs >= 0.5) == target.numpy()))
    row = model.prediction_rows(predictions)[0]
    assert set(row) == {
        "id",
        "patient_id",
        "target/effusion",
        "target/mass",
        "prob/effusion",
        "prob/mass",
    }


def test_fixed_binary_threshold_is_used_unchanged():
    model = _classifier("binary", ["normal", "pneumonia"], threshold=0.3)
    predictions = model.predict_batch(_batch(torch.tensor([0, 1, 0, 1, 1, 0])))
    assert model.calibrate(predictions) == {"threshold": 0.3, "threshold_source": "config"}
    assert model.score(predictions, model.calibrate(predictions))["threshold"] == 0.3


def test_classifier_refuses_a_detection_dataset():
    info = DatasetInfo(label_mode="detection", class_names=("blob",), in_channels=1)
    with pytest.raises(ValueError, match="classification data module"):
        ImageClassifier(nn.Identity(), info)


def test_class_weight_weights_the_multiclass_loss():
    weights = [1.0, 4.0, 2.0]
    model = _classifier("multiclass", ["a", "b", "c"], class_weight=weights)
    batch = _batch(torch.tensor([0, 1, 2, 0, 0, 1]))
    expected = nn.functional.cross_entropy(
        model(batch["image"]), batch["target"], weight=torch.tensor(weights)
    )
    assert model.training_step(batch, 0).item() == pytest.approx(expected.item())


def test_label_smoothing_pulls_binary_targets_toward_one_half():
    model = _classifier("binary", ["benign", "malignant"], label_smoothing=0.2)
    batch = _batch(torch.tensor([0, 1, 1, 0, 1, 0]))
    smoothed = batch["target"].float().reshape(-1, 1) * 0.8 + 0.1
    expected = nn.functional.binary_cross_entropy_with_logits(model(batch["image"]), smoothed)
    assert model.training_step(batch, 0).item() == pytest.approx(expected.item())


@pytest.mark.parametrize(
    ("mode", "names", "weights"),
    [("multiclass", ["a", "b", "c"], [1.0, 2.0]), ("binary", ["no", "yes"], [1.0, 2.0])],
)
def test_class_weight_must_match_a_multiclass_task(mode, names, weights):
    with pytest.raises(ValueError, match="one weight per class"):
        _classifier(mode, names, class_weight=weights)
