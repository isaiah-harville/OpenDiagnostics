"""Download PU2756 and write a manifest with the five official cross-validation folds.

    uv run python scripts/datasets/pu2756.py data/pu2756 [--delete-archives]

Source: figshare, doi:10.6084/m9.figshare.32672274 (CC BY 4.0), which needs no account.
2,756 B-mode ultrasound images of peripheral pulmonary lesions, one per patient, each with a
biopsy-confirmed benign or malignant label (Jiang et al., Sci Data 13:1711, 2026). Sizes and
MD5s are figshare's. The segmentation masks are not downloaded.

``five_fold_split.xlsx`` has a test and a train sheet for each of its folds 1 to 5. The five
test sheets partition the patients, and each train sheet holds every other patient. There is
no official validation set, so the manifest carves one out of each fold's train patients
(20%, stratified by label) and never touches that fold's test patients. Official fold ``k`` is
the manifest column ``fold{k-1}``, holding ``train``, ``val`` or ``test`` for each image.
"""

from __future__ import annotations

from pathlib import Path

from _download import (
    arguments,
    assign_splits,
    download,
    fetch_and_extract,
    relative,
    write_manifest,
)
from openpyxl import load_workbook

BASE = "https://ndownloader.figshare.com/files"
IMAGES = ("images.zip", 65547621, 830_119_533, "66f6a577968f046fc59c3a304e9b849d")
SPLITS = ("five_fold_split.xlsx", 65547615, 452_323, "643451f594a3253887b2b43375b9d51e")
FOLDS = 5
LABELS = {"Benign": "benign", "Malignant": "malignant"}
PATIENT = "Examination Serial Number"
LABEL = "Pathological Result"
VAL_FRACTION = 0.2


def _sheet(book, name: str) -> dict[str, str]:
    rows = book[name].iter_rows(values_only=True)
    header = [str(c).strip() for c in next(rows)]
    missing = {PATIENT, LABEL} - set(header)
    if missing:
        raise RuntimeError(f"sheet {name} lacks the columns {sorted(missing)}; has it changed?")
    patient, label = header.index(PATIENT), header.index(LABEL)
    return {str(r[patient]).strip(): LABELS[str(r[label]).strip()] for r in rows if r[patient]}


def read_folds(path: Path) -> tuple[dict[str, str], list[set[str]]]:
    """Each patient's label, and the test patients of official folds 1 to 5 in order."""
    book = load_workbook(path, read_only=True)
    label: dict[str, str] = {}
    tests: list[set[str]] = []
    trains: list[set[str]] = []
    for k in range(1, FOLDS + 1):
        test, train = _sheet(book, f"fold{k}_test"), _sheet(book, f"fold{k}_train")
        for patient, value in (test | train).items():
            if label.setdefault(patient, value) != value:
                raise RuntimeError(f"{path} gives patient {patient} two labels")
        if set(test) & set(train):
            raise RuntimeError(f"{path} puts a patient in both the train and test of fold {k}")
        tests.append(set(test))
        trains.append(set(train))
    for k, (test, train) in enumerate(zip(tests, trains, strict=True), 1):
        if set(label) - test != train:
            raise RuntimeError(f"fold {k} of {path} does not train on every non-test patient")
    if sum(map(len, tests)) != len(label):
        raise RuntimeError(f"the test sheets of {path} do not partition its patients")
    return label, tests


def find_images(folder: Path) -> dict[str, Path]:
    """Map each patient ID to its one image, ``images/<patient>/<n>.png``."""
    images: dict[str, Path] = {}
    for patient in sorted(p for p in folder.iterdir() if p.is_dir()):
        pngs = sorted(patient.glob("*.png"))
        if len(pngs) != 1:
            raise RuntimeError(f"{patient} holds {len(pngs)} images; PU2756 has one per patient")
        images[patient.name] = pngs[0]
    return images


def manifest_rows(
    label: dict[str, str], tests: list[set[str]], images: dict[str, Path], root: Path
) -> list[dict[str, str]]:
    if set(images) != set(label):
        missing, extra = sorted(set(label) - set(images)), sorted(set(images) - set(label))
        raise RuntimeError(
            f"the images and the split disagree: {len(missing)} patients have no image "
            f"({missing[:5]}) and {len(extra)} images have no label ({extra[:5]})"
        )
    patients = sorted(label)
    rows = [
        {"path": relative(images[p], root), "patient_id": p, "label": label[p]} for p in patients
    ]
    for fold, test in enumerate(tests):
        train = [p for p in patients if p not in test]
        carved = assign_splits(
            train, [label[p] for p in train], (1 - VAL_FRACTION, VAL_FRACTION, 0.0)
        )
        split = dict(zip(train, carved, strict=True)) | dict.fromkeys(test, "test")
        for row in rows:
            row[f"fold{fold}"] = split[row["patient_id"]]
    return rows


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    root: Path = args.root
    name, number, size, md5 = IMAGES
    fetch_and_extract(f"{BASE}/{number}", root / name, root, size, md5, args.delete_archives)
    name, number, size, md5 = SPLITS
    label, tests = read_folds(download(f"{BASE}/{number}", root / name, size, md5))
    rows = manifest_rows(label, tests, find_images(root / "images"), root)
    columns = ("path", "patient_id", "label", *(f"fold{k}" for k in range(FOLDS)))
    write_manifest(root, rows, columns)


if __name__ == "__main__":
    main()
