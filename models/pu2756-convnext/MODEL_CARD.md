# Model card: pu2756-convnext

## Summary
A ConvNeXt-S classifier that scores one B-mode lung ultrasound image of a peripheral
pulmonary lesion as malignant or benign, trained on PU2756 at 224 px under the dataset's
official five-fold cross-validation. No run has been published yet.

## Intended use
Research only. Not a medical device. It must not be used for diagnosis or to decide on a
biopsy. Every PU2756 image shows a lesion that was biopsied, so the model has never seen a
lesion-free scan and its false-positive rate in a real clinic is unknown.

## Data
- PU2756: 2,756 B-mode images, one per patient, 1,755 malignant and 1,001 benign, all
  confirmed by percutaneous biopsy. One centre (The First Affiliated Hospital of Guangzhou
  Medical University), 2020 to 2025, Esaote MyLab90 and Hitachi 850 scanners. Jiang T, et al.,
  Scientific Data 13:1711 (2026), doi:10.1038/s41597-026-07715-0. Data:
  doi:10.6084/m9.figshare.32672274.v1.
- License: the data is CC BY 4.0 (figshare). The paper itself is CC BY-NC-ND 4.0.
- Downloaded without an account by `scripts/datasets/pu2756.py`, which checks figshare's
  sizes and MD5s. The segmentation masks are not used.
- Labels: `Pathological Result` (Benign or Malignant) from `five_fold_split.xlsx`.
- **Split.** The official five folds. Their test sheets partition the patients (552, 551,
  551, 551, 551) and each fold trains on everyone else. There is no official validation set,
  so `scripts/datasets/pu2756.py` holds out 20% of each fold's training patients, stratified
  by label (seed 0), for checkpoint selection and the threshold. Official fold `k` is the
  manifest column `fold{k-1}`.

| Column | Train (benign/malignant) | Val | Test |
|---|---|---|---|
| fold0 | 640 / 1,123 | 160 / 281 | 201 / 351 |
| fold1 to fold4 | 641 / 1,123 | 160 / 281 | 200 / 351 |

## Training
Config: `config.yaml`, the `dermamnist-c-convnext` recipe. `convnext_small.fb_in22k_ft_in1k`
from timm with one input channel (the images are grayscale with tinted overlays), drop path
0.2, 224x224 px, AdamW (lr 1e-4, weight decay 0.05) with a cosine schedule over 40 epochs,
batch 64, bf16, EMA weights, label smoothing 0.1. Augmentation: random resized crop (scale 0.7
to 1) and horizontal flip; no vertical flip, since the probe is always at the top of the image.
Evaluation averages the image and its horizontal flip. The checkpoint is chosen by validation
AUROC and the threshold is Youden's J on validation.

## Results
No published run yet. The bar is the dataset paper's ConvNeXt-Tiny, AUROC 0.6950 ± 0.0272
over the five folds; see `baselines.yaml`.

| Metric | Value | 95% CI |
|---|---|---|

## Limitations and biases
- One centre, two scanner models, and a Chinese patient population.
- Images keep the scanners' on-screen text and markers, which a model can learn as a
  shortcut for scanner or settings.
- Lesion-level discrimination only: no lesion-free scans, and no clinical data.

## Reproduce
```bash
uv run python scripts/datasets/pu2756.py data/pu2756
for k in 0 1 2 3 4; do
  uv run opendiagnostics fit -c models/pu2756-convnext/config.yaml \
    --name=pu2756-convnext-fold$k --data.init_args.split_column=fold$k
done
```
