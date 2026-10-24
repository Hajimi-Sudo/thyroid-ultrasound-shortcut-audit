# Dataset Audit & Multi-Cohort Profiling

This study integrates multiple open-access thyroid nodule ultrasound datasets to establish development, external testing, and stress-testing cohorts.

## 1. Summary of Public Cohorts

| Parameter | ThyroidXL (MICCAI 2025) | TN5000 (Nature Sci Data 2025) | TN3K (MICCAI 2021) | DDTI (DataCite 2015) |
| :--- | :--- | :--- | :--- | :--- |
| **Originating Center** | Vietnam National Hospital of Endocrinology | Cancer Hospital, Chinese Academy of Med Sciences (Beijing) | Zhujiang Hospital, Southern Medical University | Universidad Nacional de Colombia |
| **Patients ($N$)** | 4,093 | Not mapped explicitly ($5,000$ images) | 2,421 | 299 |
| **Images ($N$)** | 11,635 | 5,000 | 3,493 | 347 |
| **Malignancy Label** | FNA Cytology & Surgical Pathology | FNA Biopsy & Histopathology | Histopathology | Cytology / Pathology |
| **Malignancy Rate** | 27.5% in train, 47.8% in test | 71.4% (3,572 malignant / 1,428 benign) | Variable | ~80% Benign / 20% Malignant |
| **Annotation Level** | Fine pixel-level segmentation mask | PASCAL VOC bounding boxes | Pixel-level mask | Bounding box / contour |
| **Clinical Covariates** | ACR TI-RADS scores, Age, Sex | None (strictly de-identified) | None | ACR TI-RADS, Composition, Echogenicity |
| **Ultrasound Device** | Hitachi Aloka Arietta V70 (7.5–12 MHz) | GE Logiq E9 & GE Logiq S7 (5–12, 8–15 MHz) | Mindray Resona 7, Toshiba Aplio 500 | Toshiba Nemio 17, Toshiba Aplio 500 |
| **Caliper Artifacts** | **Cleaned / None** (cropped measurement marks) | Moderate / Minimal | **Prevalent** (cross-hairs present) | **Prevalent** (cross-hairs & text present) |
| **Assigned Role** | **Primary Model Development & Internal Test** | **External Multi-Machine Validation Set** | **Caliper Artifact Stress-Test Set** | **Cross-Vendor Zero-Shot Stress Set** |

---

## 2. In-Depth Cohort Audits

### Primary Cohort: ThyroidXL
- **Accession**: HuggingFace (`VNPT-AI/ThyroidXL`) / GitHub (`VNPT-AI-Official/ThyroidXL`).
- **Official Split**:
  - *Train*: 9,541 images from 3,354 patients (2,477 benign, 877 malignant patients).
  - *Test*: 2,094 images from 739 patients (386 benign, 353 malignant patients).
- **Strengths**: Strict patient-level separation; verified lack of caliper markings on image pixels; ACR TI-RADS category labels per image; ground-truth segmentation by two board-certified radiologists with senior arbitration.

### External Validation Cohort: TN5000
- **Accession**: Figshare (`10.6084/m9.figshare.cb6a67f17c04b29e7edd19`) / Nature Scientific Data (2025).
- **Format**: Standard PASCAL VOC XML annotations with bounding boxes `[xmin, ymin, xmax, ymax]`.
- **Pre-Test Probability Bias**: 71.4% malignant prevalence reflects the tertiary cancer referral center nature of the cohort.
- **Handling Strategy**: When testing the ThyroidXL-trained model on TN5000, report both raw performance and prevalence-adjusted predictive values (PPV/NPV adjusted to community/outpatient prevalence of 10–15%).

### Stress-Testing Cohorts: TN3K & DDTI
- **Role**: Evaluate how models respond to real-world sonographer cross-hair measurement markings.
- Both datasets retain prominent calipers on a substantial fraction of images, providing a natural stress-test for shortcut vulnerability.

---

## 3. Data Integrity & Preprocessing Checklist

- [x] Convert all images to standardized RGB format and native pixel spacing.
- [x] Standardize resolution to $256 \times 256$ or $384 \times 384$ using bicubic interpolation.
- [x] Strip metadata banners, hospital watermarks, and black scanning sector boundaries to create the "Cleaned Full" condition.
- [x] Crop tight bounding boxes with 10% context margin to create the "Nodule ROI" condition.
- [x] Inpaint the segmentation mask area with background parenchymal texture using Navier-Stokes/Telea inpainting or context-matching Gaussian fill to create the "Inpainted Background" condition.
