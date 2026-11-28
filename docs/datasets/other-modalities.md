# Dermatology, pathology, retina, mammography, and MedMNIST

Part of the [dataset survey](../datasets.md).

All facts below were checked against primary pages on 2026-11-26 unless marked **(unverified)**. Access classes:

- **open**: anyone can download with no account, or with a free account and nothing else.
- **registration**: a free account plus a click-through or signed agreement (Kaggle rules, a request form, a DUA).
- **credentialed**: identity verification and training, as for PhysioNet credentialed access.

None of the datasets here requires credentialing.

Kaggle pages render with JavaScript. Kaggle rules were read in a real browser from each competition's `/rules` and `/data` tabs.

---

## Dermatology

### ISIC Archive (umbrella)

- **Name and maintainer:**
    - International Skin Imaging Collaboration (ISIC).
    - The challenge data is published by ISDIS, per the attribution "(c) by ISDIS, 2020".
    - Archive: https://www.isic-archive.com, API: https://api.isic-archive.com/api/v2/
- **Tasks and labels:**
    - Per-image clinical metadata, including a hierarchical diagnosis (`diagnosis_1`…`diagnosis_5`), `diagnosis_confirm_type`, `lesion_id`, age, sex and anatomic site.
    - The archive mixes dermoscopic and clinical images.
- **Label format:** JSON from the API, CSV from archive exports.
- **Size:** 581,002 public images, per the API `count` on 2026-11-26.
- **License and access:**
    - **The license is per image.** API facet counts: CC-BY 280,267, CC-BY-NC 247,918, CC-0 52,817. Source: https://api.isic-archive.com/api/v2/images/facets/
    - Access is **open**.
- **Image format:** JPEG, at varying resolution (each image's `pixels_x` and `pixels_y` are in its metadata).
- **Citation:** cite the challenge dataset or the contributing dataset you use (see below).
- **Pitfall:** a download mixes three licenses. Filter on `copyright_license` before redistributing or using the data commercially.

### ISIC 2019 Challenge

- **Name and maintainer:**
    - ISIC.
    - Training data aggregates HAM10000 (Medical University of Vienna), BCN_20000 (Hospital Clínic de Barcelona) and MSK.
    - Pages: https://challenge.isic-archive.com/data/ and https://challenge.isic-archive.com/landing/2019/
- **Tasks and labels:**
    - Multi-class classification with 8 training classes: melanoma, melanocytic nevus, BCC, actinic keratosis, benign keratosis, dermatofibroma, vascular lesion, SCC.
    - The test set adds an outlier class, "None of the others".
    - Metric: balanced multi-class accuracy.
- **Label format:** CSV of gold-standard diagnoses. Metadata CSV: age, sex, anatomic site and a "common lesion identifier" (training only).
- **Size:**
    - 25,331 training JPEGs and 8,238 test JPEGs.
    - **No patient ID is stated.** Only a lesion ID is given, for training images.
- **License and access:**
    - **CC BY-NC 4.0**, listed as "CC-BY-NC" on https://challenge.isic-archive.com/data/
    - The 2019 landing page itself states no license and links to "Terms of Use".
    - Access is **open**.
- **Image format:** JPEG. Resolution varies by source and is not stated on the challenge pages.
- **Citations:**
    - Tschandl P, Rosendahl C, Kittler H. HAM10000. *Sci Data* 5:180161 (2018). doi:10.1038/sdata.2018.161
    - Codella NCF et al. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 ISBI…" arXiv:1710.05006
    - Hernández-Pérez C et al. "BCN20000: Dermoscopic lesions in the wild." *Sci Data* 11:641 (2024)
- **Pitfall:**
    - HAM10000 is inside the 2019 training set, so the two must not be used as separate train and test sets.
    - There is no patient ID, so group by lesion ID at most.

### ISIC 2020 Challenge (SIIM-ISIC Melanoma Classification)

- **Name and maintainer:**
    - ISIC, with SIIM on Kaggle.
    - Image sources: Hospital Clínic de Barcelona, Medical University of Vienna, MSKCC, Melanoma Institute Australia, Univ. of Queensland, Univ. of Athens.
    - Pages: https://challenge.isic-archive.com/data/ and https://www.kaggle.com/competitions/siim-isic-melanoma-classification
- **Tasks and labels:**
    - Binary classification: `target` / `benign_malignant`, where malignant means melanoma.
    - There is also a detailed `diagnosis` field (training only).
    - Positives are 584 melanomas, 1.8% of images (paper).
- **Label format:** CSV with columns `image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target` (Kaggle data tab). The ISIC page's v2 training metadata adds a lesion ID.
- **Size:**
    - 33,126 training images from **2,056 patients**. **`patient_id` is provided.**
    - 10,982 test images. **Test labels are not public.**
    - ISIC also publishes a list of 425 duplicate images.
- **License and access:**
    - **CC BY-NC 4.0**, per https://challenge.isic-archive.com/data/ and the paper (PMC7843971). The metadata files are CC0, per the paper.
    - The Kaggle rules say: "DATA ACCESS AND USE: CC BY-NC 4.0 … Non-Commercial, Academic Research and Competition Use Only" (https://www.kaggle.com/competitions/siim-isic-melanoma-classification/rules).
    - Access is **open** from ISIC, or **registration** on Kaggle.
- **Image format:**
    - DICOM (with JPEG pixel data) or JPEG.
    - Kaggle also has TFRecords resized to 1024×1024.
    - Native size averages about 12.7 MP, ranging from 0.3 to 24 MP (paper).
- **Citations:**
    - Rotemberg V, Kurtansky N, Betz-Stablein B, et al. "A patient-centric dataset of images and metadata for identifying melanomas using clinical context." *Sci Data* 8:34 (2021). doi:10.1038/s41597-021-00815-z
    - Dataset DOI: 10.34970/2020-ds01
- **Pitfall:**
    - Melanoma prevalence is extreme (1.8%).
    - Remove the 425 listed duplicates.
    - Split by `patient_id`, since patients have many lesions.

### HAM10000

- **Name and maintainer:** "Human Against Machine with 10000 training images". Philipp Tschandl, ViDIR Group, Medical University of Vienna. Harvard Dataverse doi:10.7910/DVN/DBW86T, version 4.0: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **Tasks and labels:**
    - 7-class classification. Counts from the paper: `akiec` 327, `bcc` 514, `bkl` 1,099, `df` 115, `mel` 1,113, `nv` 6,705, `vasc` 142.
    - `dx_type` records how the diagnosis was confirmed (histo, follow_up, consensus, confocal). More than 50% are confirmed by histopathology.
    - **Segmentation:** binary lesion masks for all 10,015 images (`HAM10000_segmentations_lesion_tschandl.zip`), drawn by a single dermatologist.
- **Label format:**
    - `HAM10000_metadata` is tabular: CSV, shown as .tab on Dataverse. It has a `lesion_id` column.
    - Masks are PNG files in a zip. The PNG format is **(unverified)**.
    - The ISIC 2018 Task 3 test set ground truth is included (1,511 images, public since 2023).
- **Size:**
    - 10,015 images. There are fewer unique lesions, because some lesions have several images; the count is not stated in the paper.
    - **No patient ID is given, only `lesion_id`.**
- **License and access:**
    - **CC BY-NC 4.0.** The full "Creative Commons Attribution-NonCommercial 4.0 International Public License" text is the dataset's Terms of Use on Dataverse (API: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T).
    - The paper's Table 1 also says CC BY-NC 4.0, with a CC0 waiver for the metadata (https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/).
    - Access is **open**: no files are restricted.
- **Image format:**
    - JPEG.
    - The paper says images were cropped to 800×600 at 72 DPI. The distributed images are **600×450**, which I spot-checked through the ISIC API on ISIC_0024306 and ISIC_0034320.
- **Citation:** Tschandl P, Rosendahl C, Kittler H. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Sci Data* 5:180161 (2018). doi:10.1038/sdata.2018.161
- **Pitfall:**
    - Random image-level splits leak the same lesion across splits. Split by `lesion_id`.
    - The data is dominated by `nv` (67%).

---

## Pathology

### CAMELYON16

- **Name and maintainer:** Diagnostic Image Analysis Group, Radboud UMC, with UMC Utrecht. Page: https://camelyon16.grand-challenge.org/Data/
- **Tasks and labels:**
    - Slide-level classification: metastasis vs normal.
    - Lesion-level detection and segmentation of metastases.
- **Label format:** XML polygons (ASAP format) and WSI binary masks, per the challenge page.
- **Size:**
    - Training: 270 WSIs (160 normal, 110 with metastases), split into 170 from RUMC and 100 from UMCU.
    - Test: the challenge page says 130 WSIs. The JAMA paper evaluates **129** (49 tumor, 80 normal), and the GigaScience paper counts 399 slides in total.
    - There are no patient IDs, only slide IDs.
- **License and access:**
    - **CC0.** The CAMELYON17 data page says "This work is made available under CC0" (https://camelyon17.grand-challenge.org/Data/).
    - The paper says "licensed under the Creative Commons CC0 license" (https://pmc.ncbi.nlm.nih.gov/articles/PMC6007545/).
    - The AWS Registry lists "CC0" (https://registry.opendata.aws/camelyon/).
    - Access is **open**: GigaDB doi:10.5524/100439, AWS `s3://camelyon-dataset` with `--no-sign-request`, Google Drive or Baidu.
- **Image format:**
    - Pyramidal BigTIFF, JPEG-compressed, RGB at 8 bits per channel.
    - Pixel spacing is about 0.23–0.25 µm at level 0.
    - Scanners: 3DHistech Pannoramic 250 Flash II (RUMC) and Hamamatsu NanoZoomer-XR (UMCU).
- **Citations:**
    - Ehteshami Bejnordi B, Veta M, et al. "Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer." *JAMA* 318(22):2199–2210 (2017). doi:10.1001/jama.2017.14585
    - Litjens G, Bandi P, Ehteshami Bejnordi B, et al. "1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset." *GigaScience* 7(6):giy065 (2018). doi:10.1093/gigascience/giy065
- **Pitfall:**
    - One test slide was removed, so the test set has 129 slides.
    - The paper notes that 15 slides may contain unlabelled metastatic areas.

### CAMELYON17

- **Name and maintainer:** Radboud UMC (DIAG), with data from 5 Dutch centres (CWZ, LPON, RST, RUMC, UMCU). Page: https://camelyon17.grand-challenge.org/Data/
- **Tasks and labels:**
    - Patient-level pN-stage: pN0, pN0(i+), pN1mi, pN1, pN2.
    - Slide-level label: negative, ITC, micro or macro.
    - Lesion-level contours on 50 training slides (10 per centre). ITCs are not annotated exhaustively.
- **Label format:** XML polygons (ASAP) plus a CSV of stages. The stage CSV's format is **(unverified)**.
- **Size:**
    - 1,000 WSIs: 500 train (100 patients × 5 nodes) and 500 test.
    - **Test slides are unlabelled**: "not annotated and not labelled. A pN-stage per patient is also not given."
    - Patient IDs are encoded in filenames (e.g. `centre_0/patient_000_node_0.tif`). The paper calls these patients "artificial", grouped within one centre.
- **License and access:** **CC0** (same sources as CAMELYON16). Access is **open**.
- **Image format:** Pyramidal TIFF at 0.23–0.25 µm. The LPON slides came from a Philips scanner and were converted from iSyntax.
- **Citation:**
    - Bandi P, et al. "From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge." *IEEE Trans Med Imaging* (2019). doi:10.1109/TMI.2018.2867350 — the DOI is **(unverified)**; the paper title is listed on https://registry.opendata.aws/camelyon/
    - Also cite Litjens et al. 2018, above.
- **Pitfall:** the centres use different scanners and stains, which makes this a domain-shift benchmark. Split by patient, and report results per centre.

### PCam (PatchCamelyon)

- **Name and maintainer:** Bastiaan Veeling et al. Repo: https://github.com/basveeling/pcam
- **Tasks and labels:** binary patch classification. A patch is positive if its centre 32×32 region contains at least one tumour pixel.
- **Label format:**
    - HDF5 target arrays (`*_y.h5`).
    - Metadata CSVs (`camelyonpatch_level_2_split_{train,valid,test}_meta.csv`) give the source WSI of each patch.
- **Size:**
    - 327,680 patches: train 262,144, val 32,768, test 32,768.
    - Patches are drawn from CAMELYON16. There is no WSI overlap between splits; 20% of training WSIs were held out for validation.
- **License and access:**
    - **CC0**: "The data is provided under the CC0 License, following the license of Camelyon16." The repo code is MIT. Source: https://github.com/basveeling/pcam
    - Access is **open**: Google Drive, Zenodo record 2546921 and Baidu.
- **Image format:** 96×96 RGB patches in gzipped HDF5, taken at 10× (undersampled from 40×).
- **Citation:** Veeling BS, Linmans J, Winkens J, Cohen T, Welling M. "Rotation Equivariant CNNs for Digital Pathology." arXiv:1806.03962 (2018)
- **Pitfall:**
    - Labels depend only on the centre region, so tumour in the border does not count. Random crops or resizes can break the label semantics.
    - Use the official WSI-disjoint splits.

### PANDA (Prostate cANcer graDe Assessment)

- **Name and maintainer:** Radboud UMC and Karolinska Institutet, on Kaggle. Pages: https://panda.grand-challenge.org/data/ and https://www.kaggle.com/competitions/prostate-cancer-grade-assessment
- **Tasks and labels:**
    - Slide-level ISUP grade (0–5) and Gleason score.
    - Segmentation masks for some training slides, with values that depend on the provider:
        - Radboud: 0 background, 1 stroma, 2 benign epithelium, 3–5 Gleason 3/4/5.
        - Karolinska: 0 background, 1 benign, 2 cancer.
    - Kaggle notes that the masks may contain false positives and false negatives.
- **Label format:** CSV (`image_id, data_provider, isup_grade, gleason_score`) plus TIFF masks.
- **Size:**
    - Development (public) set: 10,616 biopsies from **2,113 patients** (paper).
    - Kaggle data: 21,135 files, 411.9 GB.
    - **No patient ID column** is in `train.csv`.
    - The validation and test sets are not public.
- **License and access:**
    - **CC BY-SA-NC 4.0**, as the organisers write it; this is the CC BY-NC-SA 4.0 license.
    - Kaggle rules: "DATA ACCESS AND USE: Competition Use, Academic Use, and Non-commercial Use - CC BY-SA-NC 4.0 license" (https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/rules).
    - The paper says "The data can be used under a Creative Commons BY-SA-NC 4.0 license" (https://pmc.ncbi.nlm.nih.gov/articles/PMC8799467/).
    - Access is **registration** (Kaggle account plus rules).
- **Image format:**
    - Multi-level TIFF.
    - Scanners: 3DHistech, Hamamatsu and Leica (paper).
    - Resolution differs by provider.
- **Citation:** Bulten W, Kartasalo K, Chen P-HC, et al. "Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge." *Nat Med* 28(1):154–163 (2022). doi:10.1038/s41591-021-01620-2
- **Pitfall:**
    - Training labels are single-pathologist and noisy.
    - Some training slides have pen marks; the test slides do not.
    - Patients have several biopsies but no patient ID, so leakage-free patient splits are impossible.

### BACH (ICIAR 2018 Grand Challenge on BreAst Cancer Histology)

- **Name and maintainer:** ICIAR 2018 challenge organisers (Aresta et al.). Pages: https://iciar2018-challenge.grand-challenge.org/Dataset/ and Zenodo https://zenodo.org/records/3632035
- **Tasks and labels:**
    - Part A: 4-class image classification (normal, benign, in situ, invasive).
    - Part B: pixel-wise WSI segmentation into the same classes.
- **Label format:**
    - Image-wise labels for the microscopy images. Their file format is **(unverified)**.
    - XML region coordinates for the WSIs.
- **Size:**
    - Microscopy: 400 training images (100 per class) and 100 test images.
    - WSIs: 30 training (10 pixel-annotated, 20 not annotated) and 10 test (arXiv:1808.04277).
    - Patient-of-origin files are only **partly** available. The page says unidentified images can be assumed to come from patients other than the identified ones.
- **License and access:**
    - **CC BY-NC-ND 4.0.** Zenodo license field: "Creative Commons Attribution Non Commercial No Derivatives 4.0 International" (https://zenodo.org/records/3632035).
    - The challenge page says "publicly available under the CC BY-NC-ND license."
    - Access is **open** from Zenodo. The challenge page also has a request form.
- **Image format:**
    - Microscopy: RGB TIFF, 2048×1536, 0.42 µm/px.
    - WSIs: SVS from a Leica SCN400 at 0.467 µm/px, e.g. 42113×62625.
- **Citation:** Aresta G, Araújo T, Kwok S, et al. "BACH: Grand challenge on breast cancer histology images." *Med Image Anal* 56:122–139 (2019). doi:10.1016/j.media.2019.05.010. arXiv:1808.04277
- **Pitfall:**
    - The ND clause bars distributing derived data, such as patches or resized copies.
    - The partial patient IDs make patient-level splits approximate.

---

## Fundus and OCT

### APTOS 2019 Blindness Detection

- **Name and maintainer:** Asia Pacific Tele-Ophthalmology Society (APTOS), on Kaggle: https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Tasks and labels:**
    - Diabetic retinopathy (DR) severity, 0–4: No DR, Mild, Moderate, Severe, Proliferative.
    - Metric: quadratic weighted kappa.
- **Label format:** CSV (`train.csv` with `id_code, diagnosis`).
- **Size:**
    - The Kaggle data tab shows 5,593 files (PNG and CSV).
    - The private test set has about 13,000 images.
    - The commonly cited 3,662 train / 1,928 public test split is consistent with 5,593 files minus 3 CSVs, but I did not see it stated. **(unverified)**
    - **No patient or eye ID.**
- **License and access:**
    - **Kaggle competition rules (non-commercial):** "You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition … and for academic research and education."
    - Redistribution is prohibited.
    - Source: https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
    - The data tab license reads "Subject to Competition Rules".
    - Access is **registration**.
- **Image format:** PNG, at varied resolutions. Images come from multiple clinics and cameras.
- **Citation:** Karthik, Maggie, Sohier Dane. "APTOS 2019 Blindness Detection." Kaggle (2019). There is no dataset paper.
- **Pitfall:**
    - Kaggle warns of noisy labels and images.
    - There are no patient IDs, and the private test set is unavailable.
    - The data cannot be redistributed, so pipelines must not mirror it.

### EyePACS (Kaggle Diabetic Retinopathy Detection, 2015)

- **Name and maintainer:** EyePACS (images), California Healthcare Foundation (sponsor), on Kaggle: https://www.kaggle.com/competitions/diabetic-retinopathy-detection
- **Tasks and labels:** DR grade 0–4, the same scale as APTOS. Metric: quadratic weighted kappa.
- **Label format:** CSV (`trainLabels.csv`).
- **Size:**
    - The data tab shows 15 files, 88.29 GB.
    - Image counts are **not stated on the Kaggle data page**. The commonly cited 35,126 train / 53,576 test is **(unverified)**.
    - **Patient IDs are in the filenames**, e.g. `1_left.jpeg`, with a left and a right image per subject.
- **License and access:**
    - **Kaggle competition rules (competition-only):** "Participants must use the Data solely for the purpose and duration of the Competition". There is no redistribution, and "Participants agree that participation … shall not be construed as having or being granted a license".
    - Source: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules
    - Access is **registration**.
- **Image format:** JPEG, high resolution, from multiple camera models. Some images are inverted.
- **Citation:** Emma Dugas, Jared, Jorge, Will Cukierski. "Diabetic Retinopathy Detection." Kaggle (2015).
- **Pitfall:**
    - **The rules restrict use to the competition's purpose and duration.** That is the most restrictive license in this list; get legal review before using it in a published project.
    - Split by patient ID so that the left and right eyes stay together.
    - Some images are flipped or inverted.

### IDRiD (Indian Diabetic Retinopathy Image Dataset)

- **Name and maintainer:** Porwal P, Pachade S, et al. (SGGS Institute, Nanded, India). Pages: IEEE DataPort https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid (doi:10.21227/H25W98) and https://idrid.grand-challenge.org/Data/
- **Tasks and labels:**
    - **Segmentation:** binary masks for 81 images. Counts: MA 81, EX 81, HE 80, SE 40; the optic disc is masked in all 81.
    - **Classification:** DR grade (International Clinical DR scale) and DME risk for all 516 images.
    - **Localization:** optic-disc and fovea centre coordinates.
- **Label format:** TIF masks and CSV grades and coordinates.
- **Size:**
    - 516 images. The grading split is 413 train / 103 test.
    - Patient IDs are **not stated**.
- **License and access:**
    - **CC BY 4.0**: "This work is licensed under a Creative Commons Attribution 4.0 International License" (https://idrid.grand-challenge.org/Data/). IEEE DataPort also shows "Creative Commons Attribution".
    - Access is **registration**: a free IEEE account login is needed to download.
- **Image format:** JPG, 4288×2848, Kowa VX-10α camera, 50° FOV, macula-centred.
- **Citation:** Porwal P, Pachade S, Kamble R, et al. "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research." *Data* 3(3):25 (2018). doi:10.3390/data3030025
- **Pitfall:**
    - It is small and comes from a single site and camera.
    - SE masks exist for only 40 images.
    - There is no patient ID to check eye pairing.

### Messidor-2

- **Name and maintainer:** LaTIM (Brest) and the Messidor program partners, distributed by ADCIS: https://www.adcis.net/en/third-party/messidor2/
- **Tasks and labels:**
    - **No official ground truth.** ADCIS says the dataset "does not contain annotations such as a diabetic retinopathy 'ground truth'".
    - Third-party grades exist but are not official.
- **Label format:** a spreadsheet of image pairings only.
- **Size:**
    - 874 examinations, 1,748 images (one image per eye).
    - Messidor-Original: 529 exams, 1,058 PNG. Messidor-Extension: 345 exams, 690 JPG.
    - The examination pairing gives patient-level grouping.
- **License and access:**
    - **Custom terms, no named license.** "Messidor-2 can be used, free of charge, for research and educational purposes. Copy, redistribution, and any unauthorized commercial use are prohibited." (https://www.adcis.net/en/third-party/messidor2/)
    - Access is **registration**: a form with personal information returns an emailed link.
- **Image format:** PNG (Original part) and JPG (Extension part). The Extension was captured on a Topcon TRC NW6, 45° FOV. The original Messidor images were 1440×960, 2240×1488 or 2304×1536.
- **Citations:**
    - Decencière E, et al. "Feedback on a publicly distributed database: the Messidor database." *Image Anal Stereol* 33(3):231–234 (2014). doi:10.5566/ias.1155
    - Abràmoff MD, Folk JC, Han DP, et al. "Automated analysis of retinal images for detection of referable diabetic retinopathy." *JAMA Ophthalmol* 131(3):351–357 (2013). doi:10.1001/jamaophthalmol.2013.1743
- **Pitfall:**
    - Labels come only from third parties, so state which label set you used.
    - The file formats are mixed.
    - Redistribution is prohibited.

### Kermany OCT and pediatric chest X-ray

- **Name and maintainer:** Daniel Kermany, Kang Zhang, Michael Goldbaum. Mendeley Data v3, doi:10.17632/rscbjbr9sj.3: https://data.mendeley.com/datasets/rscbjbr9sj/3
- **Tasks and labels:**
    - 4-class OCT classification: CNV, DME, DRUSEN, NORMAL.
    - The same deposit includes pediatric chest X-rays (pneumonia vs normal).
    - Labels come from tiered graders of increasing expertise.
- **Label format:** folder names (`train/` and `test/`, then the class).
- **Size:**
    - Training: 108,312 OCT images from **4,686 patients**: 37,206 CNV, 11,349 DME, 8,617 drusen, 51,140 normal.
    - Test: 1,000 images (250 per class) from 633 patients (Cell paper).
    - The test set contains patients independent of the training set.
    - **Patient IDs are in the filenames**: `(disease)-(randomized patient ID)-(image number)`.
- **License and access:**
    - **CC BY 4.0**, per https://data.mendeley.com/datasets/rscbjbr9sj/3. The same page says the data is "made available for use in research only", which is not a CC term and conflicts in spirit with the license.
    - Access is **open**.
- **Image format:** JPEG ("high resolution JPEG", paper), Heidelberg Spectralis SD-OCT B-scans. Sizes vary.
- **Citation:** Kermany DS, Goldbaum M, Cai W, et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell* 172(5):1122–1131.e9 (2018). doi:10.1016/j.cell.2018.02.010
- **Pitfall:**
    - Patients contribute many B-scans, so re-splitting must group by the patient ID in the filename.
    - The official test set is tiny (1,000) and balanced, which is unlike clinical prevalence.

---

## Mammography

### VinDr-Mammo

- **Name and maintainer:** Vingroup Big Data Institute (Pham HH, Nguyen Trung H, Nguyen HQ), on PhysioNet v1.0.0: https://physionet.org/content/vindr-mammo/1.0.0/
- **Tasks and labels:**
    - Breast-level BI-RADS 1–5 and ACR density A–D.
    - Detection boxes (xmin, ymin, xmax, ymax) for mass, calcification, asymmetries, architectural distortion, suspicious lymph node, skin thickening, skin retraction and nipple retraction, each with a finding BI-RADS.
    - Only BI-RADS 3–5 findings have boxes.
    - There are no masks and no pathology labels.
- **Label format:** CSV (`breast-level_annotations.csv`, `finding_annotations.csv`, `metadata.csv`), each with a `split` column.
- **Size:**
    - 5,000 four-view exams, about 20,000 images. The official split is 4,000 train / 1,000 test.
    - **There is no explicit patient ID or patient count**, only study, series and image IDs.
- **License and access:**
    - **PhysioNet Restricted Health Data License 1.5.0**, with the PhysioNet Restricted Health Data Use Agreement 1.5.0 (https://physionet.org/content/vindr-mammo/1.0.0/).
    - The Sci Data paper calls it the "Credentialed" license, but the PhysioNet page governs.
    - Access is **registration**: a PhysioNet account plus a signed DUA; no credentialing is stated.
- **Image format:** DICOM ("for presentation"; Siemens, IMS and Planmed). Resolution and bit depth are not stated; height and width are in the CSVs.
- **Citations:**
    - Nguyen HT, Nguyen HQ, Pham HH, et al. "VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography." *Sci Data* 10:277 (2023). doi:10.1038/s41597-023-02100-7. arXiv:2203.11205
    - Data: doi:10.13026/br2v-7517
- **Pitfall:**
    - Use the official split, because patient-level grouping is impossible.
    - The paper says some DICOMs are not standards-compliant.
    - BI-RADS 2 findings are unboxed.

### CBIS-DDSM

- **Name and maintainer:** Rubin lab (Stanford), hosted on TCIA: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- **Tasks and labels:**
    - Pathology (BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT), BI-RADS assessment and density.
    - Mass shape and margin, and calcification type and distribution.
    - ROI masks and crops. Detection boxes can be derived from the masks.
- **Label format:** 4 CSVs (Mass/Calc × Train/Test). ROI masks and crops are DICOM.
- **Size:**
    - 10,239 images and 6,775 studies.
    - **1,566 actual participants**, behind 6,671 "subject" IDs. Group patients by the `P_xxxxx` prefix.
    - The official split is about 80/20 by lesion type.
- **License and access:**
    - **CC BY 3.0**, plus the TCIA Data Usage Policy (https://www.cancerimagingarchive.net/collection/cbis-ddsm/).
    - Access is **open**.
- **Image format:** DICOM, converted losslessly from DDSM LJPEG via 16-bit TIFF. The source is scanned film, so resolution varies by scanner.
- **Citations:**
    - Lee RS, Gimenez F, Hoogi A, Miyake KK, Gorovoy M, Rubin DL. "A curated mammography data set for use in computer-aided detection and diagnosis research." *Sci Data* 4:170177 (2017). doi:10.1038/sdata.2017.177
    - Data: doi:10.7937/K9/TCIA.2016.7O02S9CY
- **Pitfall:**
    - The images are digitised film, not FFDM.
    - Mass masks were refined automatically (Dice 0.79 against a radiologist), and calcification ROIs are coarse.
    - Several subject IDs map to one participant.
    - The counts of removed images disagree: 339 in the paper vs 254 on TCIA.

### INbreast

- **Name and maintainer:** Breast Centre, Centro Hospitalar de São João, with INESC Porto.
    - The original site, medicalresearch.inescporto.pt, no longer resolves.
    - The current channel is a request to co-author Inês Domingues (now at CI-IPOP), who requires a signed "Transfer Agreement" (https://github.com/wentaozhu/deep-mil-for-whole-mammogram-classification/issues/12, a comment from 2025-01-30).
- **Tasks and labels:**
    - Expert contours of masses, calcifications, asymmetries and distortions, usable for detection and segmentation.
    - BI-RADS and ACR density for classification.
- **Label format:** XML contours (paper abstract).
- **Size:**
    - 115 cases, 410 images: 90 cases with 4 images and 25 mastectomy cases with 2 images.
    - The patient-ID scheme is **(unverified)**.
- **License and access:**
    - **No license is stated** in the paper abstract (Europe PMC, PMID 22078258) or on the request channel. The terms of the Transfer Agreement are not stated on https://github.com/wentaozhu/deep-mil-for-whole-mammogram-classification/issues/12
    - Access is **registration**: a manual request with a signed agreement.
    - Kaggle and torrent mirrors are unofficial.
- **Image format:** DICOM, Siemens MammoNovation FFDM. The 70 µm, 14-bit, 3328×4084 / 2560×3328 specs come from secondary papers only and are **(unverified)**.
- **Citation:** Moreira IC, Amaral I, Domingues I, Cardoso A, Cardoso MJ, Cardoso JS. "INbreast: toward a full-field digital mammographic database." *Acad Radiol* 19(2):236–248 (2012). doi:10.1016/j.acra.2011.09.014
- **Pitfall:**
    - It is tiny and has no official split.
    - Access is by personal request with unpublished terms, so it is unsuitable for CI or automated download.

### RSNA Screening Mammography Breast Cancer Detection (RSNA-SMBC, 2022–23)

- **Name and maintainer:** Radiological Society of North America. Data from Emory University (US) and BreastScreen Victoria (Australia). Kaggle: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
- **Tasks and labels:**
    - Breast-level classification: `cancer` (target), `biopsy`, `invasive`, `BIRADS` (0/1/2), `density` (A–D) and `difficult_negative_case`.
    - All labels are train only.
    - **There are no boxes or masks.**
- **Label format:** CSV (`site_id, patient_id, image_id, laterality, view, age, implant, density, machine_id, cancer, biopsy, invasive, BIRADS, prediction_id, difficult_negative_case`).
- **Size:**
    - The Kaggle data tab shows 54,713 files (314.72 GB), almost all of them training DICOMs. The hidden test set has about 8,000 patients.
    - **`patient_id` is provided.**
    - The commonly cited 54,706 images / 11,913 patients is **(unverified)**.
- **License and access:**
    - Kaggle rules: "DATA ACCESS AND USE: Non-Commercial Use, for the purposes of the competition. Re-distribution or re-identification of any data is strictly prohibited." They add that "Following the challenge's completion, there may be different revisions in licensing terms." Source: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules
    - RSNA also distributes the data under the **RSNA MIRA Dataset Research Use Agreement** (https://www.rsna.org/mira-license, V2.0 of 2025-11-05). That agreement allows research use, including research toward products, and bars redistribution and derivative works.
    - RSNA's GitHub wiki says "non-commercial purposes only" (https://github.com/RSNA/AI-Challenge-Data/wiki/RSNA-Screening-Mammography-Breast-Cancer-Detection).
    - Access is **registration**.
- **Image format:** DICOM. Many files are JPEG 2000 compressed and need extra decoders (Kaggle data tab).
- **Citations:**
    - Chen Y, Partridge GJW, Vazirabad M, et al. "Performance of Algorithms Submitted in the 2023 RSNA Screening Mammography Breast Cancer Detection AI Challenge." *Radiology* 316(2):e241447 (2025). doi:10.1148/radiol.241447
    - Data: doi:10.1148/dataset.smbc.2024
- **Pitfall:**
    - Test labels are hidden, so split `train` by `patient_id`.
    - Cancer prevalence is about 4% or lower.
    - There is a site shift.
    - The RSNA pages word the license inconsistently. Use the agreement you actually accepted.

### CMMD (The Chinese Mammography Database)

- **Name and maintainer:** Cui C, Li L, Cai H, et al., hosted on TCIA: https://www.cancerimagingarchive.net/collection/cmmd/
- **Tasks and labels:**
    - Classification only: biopsy-confirmed benign vs malignant.
    - Molecular subtype for 749 patients.
    - There are no boxes or masks.
- **Label format:** clinical XLSX.
- **Size:**
    - 1,775 patients.
    - The TCIA page gives two image counts: 5,202 in its table and "3,728" in its text.
    - **Patient IDs are present.** Subset D2 is all malignant.
- **License and access:**
    - **CC BY 4.0** (https://www.cancerimagingarchive.net/collection/cmmd/).
    - Access is **open**.
- **Image format:** DICOM, converted from TIFF, GE Senographe DS. Resolution is not stated.
- **Citations:**
    - Data: Cui C, et al. "The Chinese Mammography Database (CMMD)." TCIA (2021). doi:10.7937/tcia.eqde-4b16
    - Cai H et al. *Comput Math Methods Med* (2019). doi:10.1155/2019/2717454
- **Pitfall:**
    - There are no normal screens.
    - Splitting by subset leaks the label, because D2 is all malignant.
    - The image count is ambiguous.

---

## Ultrasound

### PU2756 (Pulmonary Ultrasound)

- **Name and maintainer:** Jiang T, Wang Q, Liang G, et al., The First Affiliated Hospital of Guangzhou Medical University, hosted on figshare: https://doi.org/10.6084/m9.figshare.32672274
- **Tasks and labels:**
    - Benign vs malignant classification of peripheral pulmonary lesions, confirmed by percutaneous biopsy.
    - One lesion mask per image.
- **Label format:** `five_fold_split.xlsx`, with a test and a train sheet for each official fold, plus age band, sex and detailed pathology.
- **Size:**
    - 2,756 B-mode PNG images, one per patient: 1,755 malignant, 1,001 benign. The images zip is 830 MB.
    - **Patient IDs are present** (`Examination Serial Number`, also the image folder name).
- **License and access:**
    - **CC BY 4.0** for the data (figshare). The paper is CC BY-NC-ND 4.0.
    - Access is **open**.
- **Image format:** RGB PNG of varying size (the paper's Table 1 summarises them), from Esaote MyLab90 and Hitachi 850 scanners with convex probes. The content is grayscale with tinted overlays.
- **Citations:**
    - Jiang T, et al. "A Pulmonary Ultrasound Dataset for Tumor Segmentation and Classification." *Sci Data* 13:1711 (2026). doi:10.1038/s41597-026-07715-0
- **Pitfall:**
    - The five official folds have no validation set; carve one from each fold's train patients.
    - Every image shows a lesion, so there are no normal scans.
    - Images keep the scanners' on-screen text.

---

## Benchmarks

### MedMNIST v2 / MedMNIST+

- **Name and maintainer:** Jiancheng Yang, Rui Shi, Bingbing Ni et al. Site: https://medmnist.com/ Code: https://github.com/MedMNIST/MedMNIST
- **Install:**
    - `pip install medmnist` (PyPI package `medmnist`, code under the Apache-2.0 license).
    - Data is hosted on **Zenodo** record 10519652, doi:10.5281/zenodo.10519652, v3.0, 2024-01-16, 60 `.npz` files, 46.1 GB. The repo calls it "the only official distribution link".
    - The per-subset CSVs mapping images to source IDs are on Google Drive.
- **Sizes:**
    - 2D subsets come at 28, 64, 128 and 224 px.
    - 3D subsets come at 28³ and 64³.
    - File names are `<subset>.npz` for size 28 and `<subset>_<size>.npz` otherwise.
- **Format:** NPZ with keys `train_images, train_labels, val_images, val_labels, test_images, test_labels`. The official train/val/test splits are fixed.
- **Task types:** classification only (multi-class, multi-label, binary, ordinal).
- **Patient IDs:** none are in the NPZ files. The paper says it splits "randomly at the patient level with a ratio of 7:1:2" where the source has no official split.
- **License and access:**
    - Zenodo and the GitHub README say: "The MedMNIST dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), except DermaMNIST under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" (https://zenodo.org/records/10519652, https://github.com/MedMNIST/MedMNIST).
    - The paper adds that "Some datasets are under CC-BY-NC license" without naming them (https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/).
    - **MedMNIST does not itself say that subsets inherit their source licenses.** That inheritance is the legal reality for derived data, and several sources are stricter than CC BY 4.0. See the table below.
    - Access is **open**.
- **Citations:**
    - Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, Pfister H, Ni B. "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification." *Sci Data* 10:41 (2023). doi:10.1038/s41597-022-01721-8. arXiv:2110.14795
    - Yang J, Shi R, Ni B. "MedMNIST Classification Decathlon." ISBI 2021. doi:10.1109/ISBI48211.2021.9434062
    - Cite the source datasets too.
- **Pitfalls:**
    - The 28 px images discard most diagnostic detail.
    - Several splits are image-level (7:1:2) on sources whose patient grouping the paper does not describe.
    - The source licenses differ from MedMNIST's blanket CC BY 4.0.
    - `medmnist/info.py` sample counts for OrganA/C/S, OrganMNIST3D and VesselMNIST3D differ slightly from the paper's Table 2: for example, OrganAMNIST train is 34,561 in the code vs 34,581 in the paper. Trust the NPZ shapes.

#### Subsets

Modality, task and counts come from the paper's Table 2 (PMC9852451), with code counts from `medmnist/info.py` where they differ. "Declared" is the `license` field in `medmnist/info.py` (https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/medmnist/info.py). "Source license" is what the original dataset's own page states.

| Subset | Source | Modality | Task | Train / Val / Test | Declared | Source license (URL) |
|---|---|---|---|---|---|---|
| PathMNIST | NCT-CRC-HE-100K + CRC-VAL-HE-7K (Kather 2019) | Colon pathology | multi-class (9) | 89,996 / 10,004 / 7,180 | CC BY 4.0 | CC BY 4.0 (https://zenodo.org/records/1214456) |
| ChestMNIST | NIH ChestX-ray14 (Wang 2017) | Chest X-ray | multi-label (14) | 78,468 / 11,219 / 22,433 | CC BY 4.0 | not stated on https://nihcc.app.box.com/v/ChestXray-NIHCC (JS-only page; could not verify) |
| DermaMNIST | HAM10000 (Tschandl 2018) | Dermatoscope | multi-class (7) | 7,007 / 1,003 / 2,005 | **CC BY-NC 4.0** | CC BY-NC 4.0 (Harvard Dataverse, above) |
| OCTMNIST | Kermany 2018 | Retinal OCT | multi-class (4) | 97,477 / 10,832 / 1,000 | CC BY 4.0 | CC BY 4.0 (https://data.mendeley.com/datasets/rscbjbr9sj/3) |
| PneumoniaMNIST | Kermany 2018 | Pediatric chest X-ray | binary | 4,708 / 524 / 624 | CC BY 4.0 | CC BY 4.0 (same Mendeley record) |
| RetinaMNIST | DeepDRiD (ISBI 2020) | Fundus | ordinal regression (5) | 1,080 / 120 / 400 | CC BY 4.0 | **CC BY-SA 4.0** (LICENSE in https://github.com/deepdrdoc/DeepDRiD) — conflicts |
| BreastMNIST | BUSI (Al-Dhabyani 2020) | Breast ultrasound | binary | 546 / 78 / 156 | CC BY 4.0 | not stated on https://scholar.cu.edu.eg/?q=afahmy/pages/dataset (behind reCAPTCHA) or in the Data in Brief article (PMC6906728; only the article is CC BY) |
| BloodMNIST | Acevedo 2020 | Blood-cell microscopy | multi-class (8) | 11,959 / 1,712 / 3,421 | CC BY 4.0 | CC BY 4.0 (https://data.mendeley.com/datasets/snkd93bnjr/1) |
| TissueMNIST | BBBC051 (Ljosa / Woloshuk) | Kidney-cortex microscopy | multi-class (8) | 165,466 / 23,640 / 47,280 | CC BY 4.0 | CC BY 3.0 Unported (https://bbbc.broadinstitute.org/BBBC051) — version differs |
| OrganAMNIST | LiTS (Bilic et al.) | Abdominal CT (axial) | multi-class (11) | 34,581 / 6,491 / 17,778 (code: 34,561 train) | CC BY 4.0 | **CC BY-NC-SA** per LiTS paper, arXiv:1901.04056; not stated on https://competitions.codalab.org/competitions/17094 — conflicts |
| OrganCMNIST | LiTS | Abdominal CT (coronal) | multi-class (11) | 13,000 / 2,392 / 8,268 (code: 12,975 / 2,392 / 8,216) | CC BY 4.0 | same as OrganAMNIST — conflicts |
| OrganSMNIST | LiTS | Abdominal CT (sagittal) | multi-class (11) | 13,940 / 2,452 / 8,829 (code: 13,932 / 2,452 / 8,827) | CC BY 4.0 | same as OrganAMNIST — conflicts |
| OrganMNIST3D | LiTS | Abdominal CT | multi-class (11) | 972 / 161 / 610 (code: 971 train) | CC BY 4.0 | same as OrganAMNIST — conflicts |
| NoduleMNIST3D | LIDC-IDRI (Armato) | Chest CT | binary | 1,158 / 165 / 310 | CC BY 4.0 | CC BY 3.0 + TCIA Data Usage Policy (https://www.cancerimagingarchive.net/collection/lidc-idri/) |
| AdrenalMNIST3D | new (MedMNIST authors) | Shape from abdominal CT | binary | 1,188 / 98 / 298 | CC BY 4.0 | MedMNIST's own statement only |
| FractureMNIST3D | RibFrac (Jin 2020) | Chest CT | multi-class (3) | 1,027 / 103 / 240 | CC BY 4.0 | **CC BY-NC 4.0** (https://zenodo.org/records/3893508, Part 1 only) — conflicts |
| VesselMNIST3D | IntrA (Yang 2020) | Shape from brain MRA | binary | 1,335 / 192 / 382 (code: 191 val) | CC BY 4.0 | not stated on https://github.com/intra3d2019/IntrA |
| SynapseMNIST3D | new (MedMNIST authors) | Electron microscopy | binary | 1,230 / 177 / 352 | CC BY 4.0 | MedMNIST's own statement only |

**Practical reading:**

- For a project that must stay commercially usable, do not assume CC BY 4.0 for these subsets:
    - DermaMNIST, the Organ* subsets and FractureMNIST3D are NC.
    - RetinaMNIST is SA.
    - ChestMNIST, BreastMNIST and VesselMNIST3D have unverified source terms.
- PathMNIST, OCTMNIST, PneumoniaMNIST, BloodMNIST, TissueMNIST, NoduleMNIST3D, AdrenalMNIST3D and SynapseMNIST3D have permissive sources. TissueMNIST and NoduleMNIST3D sources are CC BY 3.0, and NoduleMNIST3D also carries the TCIA policy.

---

## Facts not verified

- **APTOS:** the train/test image counts (3,662 / 1,928). Only the total of 5,593 files was visible.
- **EyePACS (Kaggle 2015):** the train/test image counts (35,126 / 53,576) are not on the data page.
- **HAM10000:** the number of unique lesions, and the mask file format.
- **CAMELYON17:** the IEEE TMI DOI, and the format of the stage-label file.
- **BACH:** the label file format for the microscopy images.
- **PANDA:** the image resolution per provider.
- **IDRiD:** whether patient IDs exist.
- **INbreast:** the license and Transfer Agreement terms, the image specs and the patient-ID scheme.
- **VinDr-Mammo:** the patient count and the resolution.
- **RSNA-SMBC:** the exact training image and patient counts.
- **CMMD:** the exact image count (5,202 vs 3,728) and the resolution.
- **MedMNIST sources:**
    - The ChestX-ray14 (NIH), BUSI and IntrA licenses.
    - The LiTS license, which comes from the paper, not the CodaLab page.
    - RibFrac Parts 2 and later.
