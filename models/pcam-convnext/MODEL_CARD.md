# Model card: pcam-convnext

## Summary
A ConvNeXt-Small classifier that predicts whether the centre 32x32 region of a 96x96 lymph
node histology patch contains metastatic tumour. No run is published yet.

## Intended use
Research only. Not a medical device. It must not be used to diagnose patients or to screen
slides. It was trained on patches from two Dutch centres and has not been tested on whole
slides, other scanners, other stains or other tissues.

## Data
- PatchCamelyon (PCam), Veeling et al., "Rotation Equivariant CNNs for Digital Pathology",
  MICCAI 2018 ([arXiv:1806.03962](https://arxiv.org/abs/1806.03962)). Files from
  [Zenodo record 2546921](https://zenodo.org/records/2546921), MD5-checked.
- Patches are drawn from the Camelyon16 whole-slide images (Ehteshami Bejnordi et al., JAMA
  2017) at 10x.
- License: CC0 1.0, following Camelyon16 ([PCam README](https://github.com/basveeling/pcam)).
  Weights, predictions and example images can be published.
- Split: the official split, each balanced 50/50. 20% of the Camelyon16 training slides
  were held out for validation.

  | Split | Patches | Slides |
  |---|---|---|
  | train | 262,144 | 216 |
  | validation | 32,768 | 54 |
  | test | 32,768 | 129 |

- Slide IDs: the record's metadata CSVs name the source slide of every patch, and the
  pipeline uses the slide as the patient ID. Setup checks that no slide appears in two
  splits, and the bootstrap resamples slides. The IDs are slides, not patients. The
  Camelyon16 paper (Ehteshami Bejnordi et al., JAMA 2017, Methods) reports 399 slides
  sampled from 399 patients, which implies one slide per patient, but it does not say so
  outright.
- Labels come from the pixel-level tumour annotations of Camelyon16. Tumour outside the
  centre region does not make a patch positive.

## Training
Config: `config.yaml`. ConvNeXt-Small (`convnext_small.fb_in22k_ft_in1k`, ImageNet-22k
pretrained), bf16 mixed precision, 8 epochs, batch size 64, AdamW with a cosine schedule,
weight EMA. The config sets label smoothing 0.1, but the binary loss ignored it until
`fix(modules): apply label smoothing to binary and multilabel losses`, so the published run
`20261127-054546-b1597946` trained without smoothing. A rerun on a later commit applies it.
Patches are upsampled from 96 to 224 px, the resolution the
weights were trained at; at 96 px the last stage sees a 3x3 feature map. Augmentation is a
random 90-degree rotation and a horizontal flip, which together cover all eight symmetries of
the square and keep the centre region in place. Evaluation averages the patch and its
horizontal flip. Commit, hardware and training time: from the run's `run.json`.

## Results
No published run yet. Test split, best-validation checkpoint, 95% bootstrap intervals over
the 129 test slides from the run's `metrics.json`.

| Metric | Value | 95% CI |
|---|---|---|

The decision threshold is fitted on validation with Youden's J.

## Limitations and biases
- Two centres and their scanners; no external test set.
- The intervals treat slides as independent. If a patient contributed more than one slide,
  they would be somewhat too narrow.
- Patch-level AUROC is not slide-level or lesion-level performance (the Camelyon16 metrics).

## Reproduce
```bash
uv run opendiagnostics fit -c models/pcam-convnext/config.yaml
```
