# BRISC train-test leakage audit

Question: do BRISC's train and test splits share images or patients? BRISC (Fateh et al.,
Scientific Data 13, 361, 2026; arXiv 2506.14318) says exact and near duplicates were removed
before the split, and that no obvious same-patient images cross it.

Answer: they do, heavily. 542 of the 1,000 test images have a copy in train, and 714 test
images come from a Cheng et al. patient who also has images in train. Every Cheng patient in
test is also in train. Only 184 test images (139 of them no-tumor) are free of both.

Every number below comes from a script in `scripts/` and the JSON it writes to `generated/`.

## Data and licence

| Item | Value | Source |
|---|---|---|
| BRISC copy used | Mendeley Data `c7dj4pwhcv` v3 (2026-06-23), `brisc2025.zip`, 260,175,408 bytes, SHA-256 `ce4f4699...a6f216` (matches the API) | Mendeley public API, no account |
| BRISC licence | CC BY 4.0 (Mendeley and figshare 30533120); the Mendeley record adds that third-party content may need further permission | record metadata |
| Other copies | figshare 30533120 v1 (2025-11-04), same zip size, different MD5 (`99d434e3...` vs our `928b6491...`); Kaggle `briscdataset/brisc2025` (not used); no Zenodo record found | figshare API, Zenodo search |
| Cheng set | figshare 1512427 v8, CC BY 4.0, 3,064 `.mat` slices, 233 patients, `cjdata.PID` per slice; all four zips match their published MD5 | figshare API |

The zip holds `classification_task/{train,test}/{glioma,meningioma,pituitary,no_tumor}` and
`segmentation_task/{train,test}/{images,masks}` (3,933 + 860 pairs: the tumor images). The
manifests the record describes are not in the zip. Data lives in `data/brisc/` (git-ignored).

Image counts (`generated/duplicates.json`, `images`) match the paper's Table 1:

| Class | Train | Test |
|---|---|---|
| glioma | 1,147 | 254 |
| meningioma | 1,329 | 306 |
| pituitary | 1,457 | 300 |
| no tumor | 1,067 | 140 |

The plane tags in the file names (`images_by_plane`: test 398 ax / 305 co / 297 sa) do not quite
match the paper's Table 2 (387 / 316 / 297), and 13 pixel-identical pairs carry different plane
tags (`exact_pixels.pairs_with_different_plane_tags`; for example `train_01467_me_ax` and
`train_02365_me_sa`).

How BRISC was built, per arXiv v5: start from Nickparvar's Kaggle "Brain Tumor MRI Dataset"
(7,023 images; a merge of Cheng's figshare set, SARTAJ and Br35H), keep T1 images, drop
corrupt and single-image series, remove exact and near duplicates "before any train/test
split", re-annotate masks, and separate likely same-subject images "through visual similarity
checks and metadata cross-referencing". The paper gives no deduplication method, threshold or
removal count, and says subject-level independence "cannot be guaranteed".

## Method

1. `features.py`: each image is converted to grayscale, cropped to its foreground bounding
   box and padded to a square (so border trimming and resizing do not matter). It records the
   SHA-256 of the decoded pixels and of the file, a 64-bit pHash (`imagehash`) plain and
   mirrored, 32 x 32 zero-mean unit-norm thumbnails under six variants (identity, left-right
   and up-down flips, 180 degree turn, 90% and 80% centre crops), and a DINOv2 ViT-S/14
   embedding (timm `vit_small_patch14_dinov2.lvd142m`, CPU). Same for the 3,064 Cheng slices.
2. `duplicates.py`: all 6,000 x 6,000 pairs at four levels. Thumbnail correlation is the best
   over the variants of either image.
3. `patients.py`: each BRISC image is matched to the Cheng slice with the highest thumbnail
   correlation. A match is confident at correlation >= 0.95 and a margin >= 0.02 over the best
   slice of any other patient (the runner-up from the same patient is usually an adjacent
   slice). No-tumor images cannot be in Cheng's set, so they are the null.
4. `sheets.py`: contact sheets in `generated/sheets/` (git-ignored), reviewed by eye.
5. `probe.py`: frozen-embedding probes, described under Effect.

## Duplicates (`generated/duplicates.json`)

Test images with at least one train partner, and pair counts, per level:

| Level | Test images with a train partner | glioma / meningioma / no tumor / pituitary | train-test pairs | train-train pairs | test-test pairs |
|---|---|---|---|---|---|
| identical file bytes | 7 | 0 / 2 / 0 / 5 | 9 | 41 | 5 |
| identical decoded pixels | 101 | 0 / 96 / 0 / 5 | 110 | 41 | 8 |
| thumbnail corr >= 0.99 | 542 | 66 / 269 / 1 / 206 | 600 | 1,149 | 60 |
| thumbnail corr >= 0.97 | 566 | 76 / 271 / 4 / 215 | 656 | 1,317 | 66 |
| thumbnail corr >= 0.95 | 631 | 93 / 274 / 20 / 244 | 1,050 | 2,076 | 99 |
| thumbnail corr >= 0.90 | 790 | 153 / 281 / 69 / 287 | 3,024 | 6,559 | 314 |
| pHash Hamming 0 | 399 | 37 / 210 / 0 / 152 | 448 | 791 | 46 |
| pHash Hamming <= 4 | 596 | 82 / 276 / 6 / 232 | 935 | 2,128 | 83 |
| DINOv2 cosine >= 0.98 | 493 | 62 / 235 / 19 / 177 | 574 | 1,323 | 54 |
| DINOv2 cosine >= 0.95 | 805 | 171 / 288 / 57 / 289 | 3,307 | 8,268 | 332 |

No cross-split pair at or above correlation 0.95 has different labels; 55 do at 0.90.

The 101 pixel-identical test images differ from their train copies at the byte level (only 7
are byte-identical), so a file-hash deduplication would miss them.

Verdicts by eye (sheet = the 8 strongest pairs in the band plus 8 at random):

- **corr >= 0.99 (600 pairs):** all 16 are the same image. Some are left-right mirrored
  copies (for example `train_02259_me_sa` and `test_00550_me_sa`). These are true copies.
- **0.97 to 0.99 (56 pairs):** mostly adjacent slices of one scan, a few near copies (one
  mirrored). Same scan, so patient leakage, but not copies.
- **0.95 to 0.97 (394 pairs):** adjacent or nearby slices of one scan, some mirrored.
- **0.90 to 0.95 (1,974 pairs):** mixed. Pituitary pairs are nearby slices of one scan; many
  no-tumor pairs are same-protocol slices that may be different people; one pair is glioma vs
  meningioma. Not usable as evidence alone.
- **pHash <= 4 (968 pairs):** the strongest are copies (many mirrored); the random half
  includes nearby slices. pHash agrees with the thumbnail level but adds nothing.
- **DINOv2 >= 0.98:** copies. **0.95 to 0.98:** half copies, half different slices or
  different people. Embeddings are candidates only, as planned.

The headline duplicate count is therefore **542 test images (54.2%) with a copy in train**, at
corr >= 0.99, which is the band where every pair reviewed was the same image.

Within splits the curation also left duplicates: 1,149 train-train pairs at corr >= 0.99.

## Patient leakage (`generated/patients.json`, `generated/patients.csv.gz`)

- Null: no-tumor images reach at most correlation 0.909 with any Cheng slice (99th
  percentile 0.885); none reaches 0.95. Tumor images have a median best correlation of 0.999.
- 4,333 of 4,793 tumor images match confidently (3,619 train, 714 test). The 1st percentile
  of the margin over another patient is 0.048. 4,330 agree with Cheng's label; the 3 that do
  not are BRISC pituitary vs Cheng meningioma. 4,325 of 4,333 have the same patient as the
  nearest DINOv2 neighbour, an independent check.
- Those 4,333 images map to 2,742 distinct Cheng slices; 1,544 slices are matched more than
  once, so BRISC holds about 1,591 extra copies of Cheng slices.
- **202 Cheng patients have test images, and all 202 also have train images** (228 in train,
  233 in the set).
- **714 test images (71.4% of test; 83% of the 860 tumor test images) come from those
  patients:** glioma 239 of 254, meningioma 180 of 306, pituitary 295 of 300.
- 453 of them have their exact Cheng slice in train; the rest have adjacent or other slices of
  the same patient (some in another plane).

Patient sheets (test image, matched Cheng slice, nearest train slice of that patient) were
checked by eye on pages 00 and 30: every Cheng match is the same slice, and the train image is
the same slice or the next one.

Union of copies (corr >= 0.99) and shared-patient images (`probe_corr0.99.json`,
`leaked_test_images_by_class`): **816 of 1,000 test images** (glioma 239, meningioma 279,
no tumor 1, pituitary 297). The remaining 184 are 139 no-tumor and 45 tumor images, so BRISC's test split cannot be cleaned
into a usable tumor benchmark by deletion.

## Effect (`generated/probe_corr0.99.json`)

Deleting leaked test images would leave a test set that is three quarters no-tumor, so the
comparison holds the full test split fixed and changes the training set instead. Probes on
frozen DINOv2 embeddings, trained on: `full` train; `deleaked` train (no image of a test
patient, no copy of a test image: 1,461 images, of which 1,066 no-tumor); and `random`, the
same per-class counts drawn at random from train (seed 0; 5 seeds give the spread). The
`random - deleaked` gap is the effect of leakage at equal training size and class mix.
Intervals are 95% percentile bootstraps over Cheng patients (images where none is known),
1,000 replicates.

| Probe | Metric | full | random | deleaked | random - deleaked |
|---|---|---|---|---|---|
| logistic | macro AUROC | 0.971 | 0.904 | 0.868 | +0.036 [0.027, 0.046] |
| logistic | accuracy | 0.850 | 0.535 | 0.542 | -0.007 [-0.033, 0.021] |
| 1-NN | macro AUROC | 0.970 | 0.855 | 0.785 | +0.071 [0.049, 0.090] |
| 1-NN | accuracy | 0.953 | 0.759 | 0.638 | +0.121 [0.083, 0.159] |

Across the 5 random seeds, logistic AUROC ranges 0.897 to 0.904 and 1-NN accuracy 0.737 to
0.762, so the gaps are larger than the seed spread.

Reading: a memorizing classifier (1-NN) gains 12 accuracy points from the leaked images; a
linear probe gains AUROC but not accuracy. Two limits: the deleaked train set loses not only
the test patients but also most of Cheng's images, so part of the gap may be a source shift
rather than memorization; and these are frozen-feature probes, not fine-tuned networks, whose
gain was not measured here.

## Prior work

Vangala et al., "Scores That Hold, Benchmarks That Leak" (arXiv 2610.00421, 30 Sep 2026),
audit Nickparvar, SARTAJ and Navoneel with the same Cheng-matching idea: 28.8% of Nickparvar's
test split has a near-twin in train, and 192 of 201 traceable test images share a patient with
train. They do not mention BRISC. Asare (arXiv 2611.11793) reports 36% near-duplicates on an
unnamed 7,216-image collection. BRISC re-split Nickparvar and claims to have fixed this; its
test split has 714 Cheng-traceable images, all from training patients, against 201 traceable
in Nickparvar's own test split.

## Not verified

- SARTAJ and Br35H images carry no patient IDs, so patient leakage outside Cheng's set
  (126 meningioma and all no-tumor test images) is not measured; the counts above are a floor.
- Only a sample of each sheet was reviewed by eye, not all 600 pairs at corr >= 0.99.
- The figshare and Mendeley zips differ in MD5; only the Mendeley copy was audited.
- No fine-tuned model was trained.
