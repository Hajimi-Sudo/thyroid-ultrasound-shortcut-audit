# Dataset Audit & Multi-Cohort Profiling

This study integrates multiple open-access thyroid nodule ultrasound datasets to establish development, external testing, and stress-testing cohorts.

## 1. Summary of Public Cohorts

| Parameter | ThyroidXL (MICCAI 2025) | TN5000 (Nature Sci Data 2025) | TN3K (MICCAI 2021) | DDTI (DataCite 2015) |
| :--- | :--- | :--- | :--- | :--- |
| **Originating Center** | Vietnam National Hospital of Endocrinology | Cancer Hospital, Chinese Academy of Med Sciences (Beijing) | Zhujiang Hospital, Southern Medical University | Universidad Nacional de Colombia |
| **Patients ($N$)** | 4,093 (Raw) / 3,115 (Curated Benchmark) | Not mapped explicitly ($5,000$ images) | 2,421 | 299 |
| **Images ($N$)** | 11,635 (Raw) / 3,115 (Curated Benchmark) | 5,000 | 3,493 | 347 |
| **Malignancy Label** | FNA Cytology & Surgical Pathology | FNA Biopsy & Histopathology | Histopathology | Cytology / Pathology |
| **Malignancy Rate** | 38.8% in curated benchmark (1,210 / 3,115) | 71.4% (3,572 malignant / 1,428 benign) | Variable | 32.3% (112 / 347 images) |
| **Annotation Level** | Fine pixel-level segmentation mask | PASCAL VOC bounding boxes | Pixel-level mask | Bounding box / contour |
| **Clinical Covariates** | ACR TI-RADS scores, Age, Sex | None (strictly de-identified) | None | ACR TI-RADS, Composition, Echogenicity |
| **Ultrasound Device** | Hitachi Aloka Arietta V70 (7.5–12 MHz) | GE Logiq E9 & GE Logiq S7 (5–12, 8–15 MHz) | Mindray Resona 7, Toshiba Aplio 500 | Toshiba Nemio 17, Toshiba Aplio 500 |
| **Caliper Artifacts** | **Cleaned / None** (cropped measurement marks) | Moderate / Minimal | **Prevalent** (cross-hairs present) | **Prevalent** (cross-hairs & text present) |
| **Assigned Role** | **Primary Model Development & Internal Test** | **External Multi-Machine Validation Set** | **Caliper Artifact Stress-Test Set** | **Cross-Vendor Zero-Shot Stress Set** |

---

## 2. In-Depth Cohort Audits

### Primary Cohort: Curated ThyroidXL Benchmark ($N = 3,115$)
- **Accession**: HuggingFace (`VNPT-AI/ThyroidXL`) / GitHub (`VNPT-AI-Official/ThyroidXL`).
- **Official Raw Archive**:
  - 11,635 images across 4,093 patients (2,863 raw benign, 1,230 raw malignant).
- **Four-Stage Rigorous Attrition Audit**:
  - *Stage 1*: Exclusion of 671 indeterminate cytology cases lacking definitive surgical pathology (659 benign [Bethesda I/III/IV], 12 malignant [Bethesda V without post-op histology]).
  - *Stage 2*: Exclusion of 188 confluent multinodular goiters or unsegmented lesions (184 benign, 4 malignant).
  - *Stage 3*: Exclusion of 72 severe Doppler overlays or macrocalcification acoustic drop-outs (70 benign, 2 malignant).
  - *Stage 4*: Exclusion of 47 cine sweep duplicates lacking primary cross-sectional plane (45 benign, 2 malignant).
  - *Total Excluded*: 978 patients (958 benign, 20 malignant).
- **Final Verified Benchmark ($N = 3,115$)**:
  - $1,905$ confirmed benign and $1,210$ confirmed malignant examinations.
- **Zero-Leakage Patient-Level Split (Stratified Random Hashing)**:
  - *Training Partition (80%)*: $N = 2,492$ unique patients ($1,547$ benign, $945$ malignant).
  - *Internal Test Partition (20%)*: $N = 623$ unique patients ($358$ benign, $265$ malignant).
  - *Leakage Audit*: Verified $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$, guaranteeing complete patient-level independence between training and testing.

### External Validation Cohort: DDTI ($N = 347$ images, $299$ patients)
- **Accession**: Universidad Nacional de Colombia / Hospital de San José.
- **Composition**: 347 images across 299 patients (235 benign images / 207 benign patients, 112 malignant images / 92 malignant patients).
- **Role**: Independent multi-scanner evaluation under authentic cross-hardware domain shift (Hitachi Aloka $\to$ Toshiba) and authentic sonographer measurement crosshairs present on 100% of scans.
- **Statistical Handling**: Analyzed at the image level using patient-clustered bootstrap resampling (2,000 iterations) to account for intra-patient correlation, and aggregated at the patient level using mean and maximum probability pooling.

### Multi-Vendor Cross-Center Generalization Cohort: TN5000
- **Accession**: Figshare (`10.6084/m9.figshare.cb6a67f17c04b29e7edd19`) / Nature Scientific Data (2025).
- **Format**: Standard PASCAL VOC XML annotations with bounding boxes `[xmin, ymin, xmax, ymax]`.
- **Pre-Test Probability Bias**: 71.4% malignant prevalence reflects the tertiary cancer referral center nature of the cohort.
- **Handling Strategy**: Report both raw performance and prevalence-adjusted predictive values (PPV/NPV adjusted to outpatient prevalence of 10–15%).

---

## 3. Data Integrity & Preprocessing Checklist

- [x] Convert all images to standardized RGB format and native pixel spacing.
- [x] Standardize resolution to $224 \times 224$ using bicubic interpolation.
- [x] Strip metadata banners, hospital watermarks, and black scanning sector boundaries to create Condition 2 (Cleaned Full-Frame).
- [x] Inpaint the segmentation mask area with background parenchymal texture using fast marching to create Condition 3 (Inpainted Background).
- [x] Counterfactually inject clinical 3-pixel electronic calipers onto both benign and malignant test cases to create Condition 4 (Caliper-Stressed).
- [x] Crop tight bounding boxes with 10% anatomical margin to create Condition 5 (Nodule ROI-Constrained).
