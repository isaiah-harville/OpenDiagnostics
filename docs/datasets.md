# Datasets

Which datasets have a published model, which are still to do, and the survey behind the
choice: licenses, access terms, patient IDs, and the published results each model has to
beat.

## Done: published models

Every dataset here has a published model. Each row links the model, its latest published
run (metrics with 95% intervals, curves, examples, and weights), and its paper, when there
is one.

--8<-- "notebook/published-table.snippet"

All of these are [MedMNIST v2](datasets/other-modalities.md#medmnist-v2-medmnist) subsets at
224 px:

* **What it is:** 18 subsets (12 2D, 6 3D) of 2D images at 28 to 224 px and 3D volumes at
  28³ and 64³. It ships fixed official splits and is classification only.
* **Access:** open, from Zenodo record 10519652. This repository reads the `.npz` files
  directly and checks each against the published MD5, so it needs no `medmnist` package.
* **License caveat:** MedMNIST declares every subset CC BY 4.0, except DermaMNIST
  (CC BY-NC 4.0). Several source datasets are stricter: RetinaMNIST's source is CC BY-SA
  4.0, and the Organ\* and FractureMNIST3D sources are non-commercial. PneumoniaMNIST,
  the worked example here, comes from a CC BY 4.0 source (Kermany et al.), so the two
  agree. See the [per-subset table](datasets/other-modalities.md#subsets).
* **No patient IDs:** the files carry none, so leakage cannot be checked. Treat MedMNIST
  results as pipeline benchmarks.

## To do

Datasets without a published model yet. The full-resolution sources of several MedMNIST
subsets are here too: NIH ChestX-ray14 (ChestMNIST), Kermany OCT and pediatric chest X-ray
(OCTMNIST, PneumoniaMNIST), and HAM10000 (DermaMNIST). Their MedMNIST versions are done;
the full-resolution data, with patient-level splits, is not.

### Next up

We ranked candidates on five criteria, in order:

1. Anyone can reproduce the result without credentialing.
2. The license allows publishing weights and results in an open-source repository.
3. Patient IDs exist, so splits can be shown leakage-free.
4. The test labels come from experts, not NLP.
5. There is enough data to give tight confidence intervals.

#### Classification

1. **NIH ChestX-ray14.**
   * Open, with the most permissive terms of any large chest X-ray set ("no
     restrictions", with attribution).
   * 112,120 images from 30,805 patients, with patient IDs and an official patient-level
     split.
   * The training labels are NLP-mined and noisy. Google's adjudicated expert labels
     (all findings on 810 test images, and 4 findings on 4,374 images) give an expert
     reference test set, which is what lets us report honest numbers.
   * It loads directly through the `csv_classification` dataset (PNG plus
     `Data_Entry_2017.csv`).
2. **Kermany OCT, then the full-resolution pediatric chest X-rays.**
   * CC BY 4.0, open, patient IDs in the filenames, and labels from tiered expert graders.
   * The OCT test set is patient-independent from training.
   * It extends the PneumoniaMNIST example to full resolution and patient-level splits.
     The official test set is small (1,000 images, balanced), so add a patient-level
     split of the training data for tighter intervals.
3. **CheXpert, as external validation for model 1.**
   * Its radiologist-consensus validation and test sets (200 and 500 studies) are the
     strongest public chest X-ray reference standard.
   * It comes from a different institution, so it measures cross-site generalization.
   * Registration and non-commercial research terms mean we would publish metrics, not
     redistribute data. Check the agreement before releasing any weights trained on it.
4. **ISIC 2020.**
   * CC BY-NC 4.0, open, and `patient_id` is provided.
   * Melanoma is 1.8% of images, a realistic test of reporting AUPRC and calibration
     next to AUROC.
   * It adds dermatology. The non-commercial license limits downstream use.

Not chosen first:

* **MIMIC-CXR and VinDr-CXR** need PhysioNet credentialing.
* **PadChest** labels are 73% model-generated.
* **APTOS and EyePACS** have Kaggle terms that forbid redistribution or limit use to the
  competition.

#### Detection

1. **RSNA Pneumonia Detection Challenge.**
   * Open download, with terms that allow commercial and non-commercial use with
     attribution.
   * 30,000 radiographs with radiologist-drawn boxes. 4,527 cases were triple-read.
   * It is 2D DICOM with one box class, a direct fit for the `coco_detection` dataset and
     torchvision Faster R-CNN.
   * One requirement: the images are a subset of NIH ChestX-ray14. Split through the
     RSNA-to-NIH mapping by NIH patient ID, and do not evaluate a model pretrained on NIH
     images on overlapping patients.
2. **CBIS-DDSM.**
   * CC BY 3.0, open, with pathology-verified mass and calcification ROIs (boxes derive
     from the masks).
   * 1,566 participants, grouped by the `P_xxxxx` prefix.
   * It adds mammography. The images are digitized film, not digital mammography, so
     results do not transfer to modern screening.
3. **VinDr-CXR**, the best chest X-ray detection set.
   * 22 box classes, 3 readers per training image, and a 5-reader consensus test set.
   * It needs PhysioNet credentialing and has no patient IDs, so only the official split
     is valid.
   * Worth doing after model 1, with the user's go-ahead to apply for access.
4. **LUNA16**, the reference benchmark for FROC and CPM.
   * CC BY 4.0, open, 888 CT scans.
   * It needs 3D detection, which is on the [roadmap](design.md#roadmap) and not yet
     supported. Its 10 official cross-validation subsets must be used as published.

Not chosen first:

* **DeepLesion:** the primary source states no license.
* **VinDr-Mammo:** PhysioNet restricted access, and no patient IDs.
* **Duke Breast MRI and PI-CAI:** 3D, and CC BY-NC.
* **NIH ChestX-ray14's own boxes:** only 983 images.


### Everything else we surveyed

"Patient IDs" says whether a patient-level split is possible, which this pipeline needs to
rule out leakage. Counts are as stated by the primary source; where sources disagree, the
detail page gives both. Each dataset links to its detail entry, with its labels, file
format, and sources.

#### Chest X-ray

| Dataset | Task | Size | Patient IDs | License | Access |
|---|---|---|---|---|---|
| [NIH ChestX-ray14](datasets/chest-xray.md#nih-chestx-ray14-chestx-ray8) | multilabel cls; 983 images with boxes | 112,120 / 30,805 | yes | "no restrictions", attribution required | open |
| [CheXpert](datasets/chest-xray.md#chexpert) | multilabel cls | 224,316 / 65,240 | yes | Stanford research use agreement, non-commercial | registration |
| [CheXpert Plus](datasets/chest-xray.md#chexpert-plus) | cls, report generation | 223,228 / 64,725 | yes | Stanford University Dataset Research Use Agreement | registration |
| [MIMIC-CXR(-JPG)](datasets/chest-xray.md#mimic-cxr-jpg) | multilabel cls, reports | 377,110 / 65,379 | yes | PhysioNet Credentialed Health Data License 1.5.0 | credentialed |
| [RSNA Pneumonia](datasets/chest-xray.md#rsna-pneumonia-detection-challenge-2018) | **detection** | 30,000 / 12,274 | via NIH mapping | RSNA terms: commercial or non-commercial use with attribution | open |
| [VinDr-CXR](datasets/chest-xray.md#vindr-cxr) | **detection** and cls | 18,000 / not given | **no** | PhysioNet Credentialed Health Data License 1.5.0 | credentialed |
| [PadChest](datasets/chest-xray.md#padchest) | multilabel cls | 160,868 / 67,625 | yes | PADCHEST Dataset Research Use Agreement | registration |
| [SIIM-ACR Pneumothorax](datasets/chest-xray.md#siim-acr-pneumothorax-segmentation-2019) | segmentation | 15,257 exams | not stated | not stated (Kaggle rules unverified) | registration |

#### MRI

| Dataset | Task | Size | Patient IDs | License | Access |
|---|---|---|---|---|---|
| [BraTS (2026)](datasets/mri-ct.md#brats-brain-tumors-2026-edition-and-lineage) | segmentation; some cls tasks | 2,040 patients (2021); METS 2026: 1,778 cases | yes | CC BY-NC 4.0 | registration |
| [fastMRI](datasets/mri-ct.md#fastmri-knee-brain-prostate-breast) | reconstruction; fastMRI+ boxes | knee ≈1,500 raw, brain ≈7,000 | not stated | NYU fastMRI Dataset Sharing Agreement | credentialed |
| [OASIS-1/2/3](datasets/mri-ct.md#oasis-oasis-1-oasis-2-oasis-3-oasis-4-noted) | cls, regression | OASIS-3: 1,378 participants | yes | OASIS Data Use Agreement, non-commercial | credentialed |
| [ADNI](datasets/mri-ct.md#adni-alzheimers-disease-neuroimaging-initiative) | cls, progression | 2,470 participants | yes | ADNI Data Use Agreement, no redistribution | credentialed |
| [UK Biobank imaging](datasets/mri-ct.md#uk-biobank-imaging) | population cohort | target 100,000 | per-application IDs | UK Biobank Access Procedures and MTA, fee | credentialed |
| [Duke Breast Cancer MRI](datasets/mri-ct.md#duke-breast-cancer-mri-tcia) | 3D detection, seg, cls | 922 patients | yes | CC BY-NC 4.0 | open |
| [ACDC](datasets/mri-ct.md#acdc-automated-cardiac-diagnosis-challenge) | seg and 5-class cls | 150 patients | yes | CC BY-NC-SA 4.0 | open |
| [PI-CAI](datasets/mri-ct.md#pi-cai-prostate-imaging-cancer-ai) | detection and diagnosis | 1,500 cases / 1,476 patients | yes | CC BY-NC 4.0 | open |

#### CT

| Dataset | Task | Size | Patient IDs | License | Access |
|---|---|---|---|---|---|
| [LIDC-IDRI](datasets/mri-ct.md#lidc-idri) | detection, seg, cls | 1,018 scans / 1,010 patients | yes | CC BY 3.0 | open |
| [LUNA16](datasets/mri-ct.md#luna16) | **detection** (points), FP reduction | 888 scans | via LIDC | CC BY 4.0 | open |
| [DeepLesion](datasets/mri-ct.md#deeplesion) | 2D detection | 32,735 lesions / 4,427 patients | yes | **not stated** by the primary source | open |
| [KiTS23](datasets/mri-ct.md#kits23-kidney-tumor-segmentation-challenge-2023) | segmentation | 599 cases | yes | CC BY-NC-SA 4.0 | open |
| [Medical Segmentation Decathlon](datasets/mri-ct.md#medical-segmentation-decathlon-msd) | segmentation, 10 tasks | 2,633 images | not stated | CC BY-SA 4.0 | open |
| [TotalSegmentator](datasets/mri-ct.md#totalsegmentator-ct-dataset-mr-dataset-noted) | segmentation | 1,228 CT | not stated | CC BY 4.0 | open |
| [RSNA Intracranial Hemorrhage](datasets/mri-ct.md#rsna-intracranial-hemorrhage-detection-2019) | slice-level cls | 25,312 exams / 874,035 images | patient-disjoint sets | RSNA terms, non-commercial | registration |

#### Dermatology, pathology, retina, mammography

| Dataset | Task | Size | Patient IDs | License | Access |
|---|---|---|---|---|---|
| [ISIC 2020](datasets/other-modalities.md#isic-2020-challenge-siim-isic-melanoma-classification) | binary cls | 33,126 / 2,056 patients | yes | CC BY-NC 4.0 | open |
| [ISIC 2019](datasets/other-modalities.md#isic-2019-challenge) | 8-class cls | 25,331 train | lesion ID only | CC BY-NC 4.0 | open |
| [HAM10000](datasets/other-modalities.md#ham10000) | 7-class cls, seg | 10,015 | lesion ID only | CC BY-NC 4.0 | open |
| [CAMELYON16/17](datasets/other-modalities.md#camelyon16) | WSI cls, detection | 399 / 1,000 WSIs | slide; patient (C17) | CC0 | open |
| [PCam](datasets/other-modalities.md#pcam-patchcamelyon) | binary patch cls | 327,680 patches | WSI-disjoint splits | CC0 | open |
| [PANDA](datasets/other-modalities.md#panda-prostate-cancer-grade-assessment) | WSI grading | 10,616 biopsies / 2,113 patients | **no** | CC BY-NC-SA 4.0 | registration |
| [Kermany OCT and pediatric CXR](datasets/other-modalities.md#kermany-oct-and-pediatric-chest-x-ray) | 4-class OCT, binary CXR | 109,312 OCT / 5,319 patients | yes (filenames) | CC BY 4.0 | open |
| [IDRiD](datasets/other-modalities.md#idrid-indian-diabetic-retinopathy-image-dataset) | DR grade, lesion seg | 516 | not stated | CC BY 4.0 | registration |
| [APTOS 2019](datasets/other-modalities.md#aptos-2019-blindness-detection) | DR grade | ≈5,590 | no | Kaggle rules, non-commercial, no redistribution | registration |
| [EyePACS (Kaggle 2015)](datasets/other-modalities.md#eyepacs-kaggle-diabetic-retinopathy-detection-2015) | DR grade | ≈88 GB | yes (filenames) | Kaggle rules: competition use only | registration |
| [Messidor-2](datasets/other-modalities.md#messidor-2) | images only | 1,748 | exam pairing | custom: research only, no redistribution | registration |
| [VinDr-Mammo](datasets/other-modalities.md#vindr-mammo) | **detection**, BI-RADS, density | 5,000 exams | **no** | PhysioNet Restricted Health Data License 1.5.0 | registration |
| [CBIS-DDSM](datasets/other-modalities.md#cbis-ddsm) | cls, seg (boxes from masks) | 10,239 images / 1,566 participants | yes (prefix) | CC BY 3.0 | open |
| [INbreast](datasets/other-modalities.md#inbreast) | detection, seg | 410 images / 115 cases | not verified | not stated; signed transfer agreement | registration |
| [RSNA Screening Mammography](datasets/other-modalities.md#rsna-screening-mammography-breast-cancer-detection-rsna-smbc-202223) | breast-level cls | ≈54,700 files | yes | Kaggle rules, non-commercial; RSNA MIRA agreement | registration |
| [CMMD](datasets/other-modalities.md#cmmd-the-chinese-mammography-database) | benign/malignant cls | 1,775 patients | yes | CC BY 4.0 | open |

#### Ultrasound

| Dataset | Task | Size | Patient IDs | License | Access |
|---|---|---|---|---|---|
| [PU2756](datasets/other-modalities.md#pu2756-pulmonary-ultrasound) | benign/malignant cls, seg | 2,756 images / 2,756 patients | yes, one image each | CC BY 4.0 | open |

## The bar to beat

The published results a new model has to beat on each shortlisted benchmark, under that
benchmark's protocol. The [benchmarks page](benchmarks.md) gives the source of every number,
whether the test labels are public, and the protocol pitfalls that make many published
numbers incomparable.

| Benchmark | Metric | Best published | By | Test labels public |
|---|---|---|---|---|
| PneumoniaMNIST | test AUC / ACC | 0.997 / 0.966 (preprint); 0.995 / 0.961 (MedViT-S); 0.9892 / 0.9156 (mean of 3 seeds) | C-Mixer; MedViT-S; MedMNIST+ DINO ViT-B | yes |
| NIH ChestX-ray14, official split | mean AUROC, 14 findings | 0.833 (self-reported), 0.828 (rerun by another group) | EVA-X-S | yes |
| CheXpert, radiologist test set | mean AUROC, 5 findings | 0.930 (ensemble) | DeepAUC-v1 | yes, competition closed |
| RSNA Pneumonia | Kaggle private mAP | 0.25475 | Ian Pan and Alexandre Cadrin | late submissions on Kaggle |
| VinDr-CXR (VinBigData) | Kaggle private mAP@0.4 | 0.314 | ℳS²Ƒ | via PhysioNet, credentialed |
| ISIC 2020 | Kaggle private AUROC | 0.9490 (used ISIC 2018 and 2019 data) | All Data Are Ext | no, scored by submission |
| CBIS-DDSM, official split | two-view AUROC | 0.8643 (0.8658 with test-time augmentation) | arXiv 2503.19945 | yes |
| LUNA16, 10-fold CV | CPM | 0.951 to 0.952 | PAtech; Liu et al. 2019 | yes, self-reported CV |
| Kermany OCT, 1,000-image test | accuracy | 99.80% | OpticNet-71 | yes |
| PU2756, official 5-fold CV | mean test AUROC | 0.6950 ± 0.0272 | ConvNeXt-Tiny (dataset paper) | yes |

A benchmark whose test labels are public invites test-set tuning, so a win there counts
only with selection on validation, means over seeds, and the code to rerun it.


## Leakage traps to remember

* RSNA Pneumonia and SIIM-ACR reuse NIH ChestX-ray14 images. CheXpert Plus reuses CheXpert
  images. LUNA16 is a subset of LIDC-IDRI. HAM10000 is inside ISIC 2019. The MSD brain task
  is BraTS 2016/17, and the MSD liver task is LiTS. Never put a dataset and its parent on
  opposite sides of a train/test split.
* ISIC 2019 and HAM10000 give only lesion IDs, and VinDr-CXR, VinDr-Mammo, PANDA, and
  MedMNIST give no patient IDs at all. Use the official splits, and report that
  patient-level leakage could not be checked.
* Longitudinal cohorts (OASIS, ADNI, UK Biobank) and multi-image studies (MIMIC-CXR,
  CheXpert, NIH, Kermany OCT) must be split by subject.

## How the survey was checked

**How it was checked.**

* Every fact was checked on 2026-11-26 against a primary source: the dataset's official
  page, its host (PhysioNet, TCIA, Synapse, Zenodo, grand-challenge, Kaggle rules), or the
  dataset paper.
* Licenses are quoted as the primary source names them. Where a source names none, the
  entry says "not stated" and gives no guess.
* Each detail file ends with the facts that could not be verified.
* This is a research summary, not legal advice. Read the agreement you accept before
  publishing weights or predictions.

**Access classes.**

* **Open**: download with no account, or a free account with click-through terms only.
* **Registration**: an account plus an agreement or request form.
* **Credentialed**: identity verification, training, or a reviewed application or DUA
  (for example PhysioNet credentialing, ADNI, fastMRI).

**Details**, with citations and source links for every claim:

* [Chest X-ray](datasets/chest-xray.md)
* [MRI and CT](datasets/mri-ct.md)
* [Dermatology, pathology, fundus and OCT, mammography, and MedMNIST](datasets/other-modalities.md)
