import csv
import hashlib
import importlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PU2756_PATIENTS = [f"PX{i:02d}" for i in range(40)]


@pytest.fixture
def pu(root, monkeypatch):
    monkeypatch.syspath_prepend(str(root / "scripts/datasets"))
    return importlib.import_module("pu2756")


class Routes:
    """A fake web server: one body per URL, and a log of the URLs requested."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.requested: list[str] = []

    def __call__(self, request):
        self.requested.append(request.full_url)
        return io.BytesIO(self.bodies[request.full_url])


def _pu2756_split(path: Path, pu, extra_train: str | None = None) -> None:
    """Five folds of eight test patients each; ``extra_train`` also joins fold 1's train."""
    from openpyxl import Workbook

    book = Workbook()
    book.remove(book.active)
    for k in range(1, 6):
        test = PU2756_PATIENTS[k - 1 :: 5]
        train = [p for p in PU2756_PATIENTS if p not in test or (k == 1 and p == extra_train)]
        for part, members in (("test", test), ("train", train)):
            sheet = book.create_sheet(f"fold{k}_{part}")
            sheet.append(["Age", "Gender", pu.PATIENT, pu.LABEL, "Detailed Pathological Result"])
            for p in members:
                value = ["Benign", "Malignant"][PU2756_PATIENTS.index(p) % 3 > 0]
                sheet.append(["50-54", "Male", p, value, value])
    book.save(path)


def test_pu2756_split_with_a_patient_in_a_fold_train_and_test_is_an_error(pu, tmp_path):
    _pu2756_split(tmp_path / "split.xlsx", pu, extra_train="PX00")
    with pytest.raises(RuntimeError, match="both the train and test of fold 1"):
        pu.read_folds(tmp_path / "split.xlsx")


def test_pu2756_script_writes_the_official_folds_and_carves_val_from_train(
    pu, tmp_path, monkeypatch
):
    _pu2756_split(tmp_path / "split.xlsx", pu)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        for i, p in enumerate(PU2756_PATIENTS):
            image = io.BytesIO()
            Image.fromarray(np.full((8, 10, 3), 90, np.uint8)).save(image, format="PNG")
            handle.writestr(f"images/{p}/{i % 4}.png", image.getvalue())
    bodies = {
        "images.zip": archive.getvalue(),
        "split.xlsx": (tmp_path / "split.xlsx").read_bytes(),
    }
    server = Routes({f"{pu.BASE}/{n}": body for n, body in enumerate(bodies.values())})
    for n, (constant, (name, body)) in enumerate(
        zip(("IMAGES", "SPLITS"), bodies.items(), strict=True)
    ):
        monkeypatch.setattr(pu, constant, (name, n, len(body), hashlib.md5(body).hexdigest()))
    monkeypatch.setattr("urllib.request.urlopen", server)
    root = tmp_path / "pu2756"
    monkeypatch.setattr("sys.argv", ["pu2756.py", str(root)])
    pu.main()
    with (root / "manifest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert rows[1]["path"] == "images/PX01/1.png"
    for k in range(5):
        assert {r["patient_id"] for r in rows if r[f"fold{k}"] == "test"} == set(
            PU2756_PATIENTS[k::5]
        )
        val = [r["label"] for r in rows if r[f"fold{k}"] == "val"]
        assert len(val) == 7
        assert set(val) == {"benign", "malignant"}
    pu.main()
    assert len(server.requested) == 2
