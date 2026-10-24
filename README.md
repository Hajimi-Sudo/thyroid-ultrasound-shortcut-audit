# Beyond Diagnostic Accuracy: Clinical Audit of Shortcut Learning and Non-Lesion Confounders in Deep Learning Models for Thyroid Nodule Ultrasound Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Peer Review: Under Review](https://img.shields.io/badge/Peer%20Review-Under%20Review-blue.svg)]()
[![Domain: Medical AI & Sonography](https://img.shields.io/badge/Domain-Medical%20AI%20%26%20Sonography-0072B2.svg)]()
[![Status: Code Release](https://img.shields.io/badge/Status-Code%20Release-success.svg)]()

This repository contains the official open-source evaluation suite, perturbation pipelines, multi-architecture audit models, and decision-support analytics for investigating shortcut learning, non-lesion artifact memorization, and cross-dataset generalizability in deep learning systems for thyroid nodule ultrasound analysis.

---

## 📌 Executive Summary & Key Findings

While deep learning models frequently achieve outstanding discrimination on curated thyroid ultrasound benchmarks ($\text{AUROC} > 0.90$), their real-world clinical adoption remains constrained by vulnerability to non-lesion confounders. In this study, we perform a systematic clinical and algorithmic audit across convolutional (ResNet-50, ConvNeXt-Tiny) and vision transformer (Swin-Transformer Tiny) architectures:

1. **Prediction Hijacking by Sonographer Calipers**:
   - Injecting standard measurement crosshairs (+) onto benign nodules elevates the mean predicted malignancy risk by $+0.312 \pm 0.089$ ($p < 10^{-15}$, Wilcoxon signed-rank test).
   - This diagnostic shift degrades model specificity from $88.5\%$ to $36.2\%$, converting true benign nodules into false-positive alerts that trigger unnecessary fine-needle aspiration (FNA) biopsies.
2. **Attribution Energy Leakage**:
   - Feature attribution analysis reveals that $41.2\% \pm 5.8\%$ ($95\%\text{ CI}: 40.7\% - 41.7\%$) of Grad-CAM gradient energy concentrates on synthetic crosshairs rather than true suspicious sonographic features (microcalcifications, irregular margins, or hypoechogenicity).
3. **Acoustic Background Shortcut Memorization**:
   - Inpainting probe models trained exclusively on background parenchyma (with the entire nodule occluded) still achieve an $\text{AUROC}$ of $0.814$, proving that ambient speckle texture and scanner-specific margins leak substantial diagnostic information.
4. **Clinical Mitigation via ROI Constraints**:
   - Constraining the model input to a nodule-focused region of interest (ROI) with a $10\%$ anatomical margin eliminates caliper sensitivity, improves external generalizability on the DDTI cohort ($\text{AUROC } 0.864$ vs. $0.782$ for raw full-frame, $p = 0.003$), and achieves a $44.8\%$ benign biopsy avoidance rate at the pre-specified safety target ($\text{Sensitivity} \ge 95.0\%$).

---

## 🔬 Methodological Framework

The audit investigates five controlled representation conditions:

| Audit Condition | Input Representation | Experimental Scope & Objective |
| :--- | :--- | :--- |
| **Condition 1: Raw Full-Frame** | Full uncropped B-mode sonogram ($224 \times 224$) | Establishes the unconstrained baseline susceptibility to peripheral vendor watermarks, metadata, and background texture. |
| **Condition 2: Cleaned Full-Frame** | Peripheral margins cropped ($5\%$ left/right/bottom, $8\%$ top) | Evaluates whether removing external text/scales mitigates non-lesion bias while preserving the acoustic background. |
| **Condition 3: Inpainted Background** | Central nodule masked and filled with parenchymal texture | Quantifies shortcut capacity by determining diagnostic accuracy when the nodule is entirely hidden from the AI. |
| **Condition 4: Caliper-Stressed** | Synthetic measurement calipers injected onto test images | Stress-tests whether operator measurement markers hijack the classifier into predicting false-positive malignancy. |
| **Condition 5: Nodule ROI-Constrained** | Bounding box cropped with $10\%$ anatomical margin | Forces the model to extract features exclusively from nodular tissue and immediate interface boundaries. |

---

## 📊 Dataset Architecture & Roles

This study utilizes three standardized, pathology-confirmed thyroid ultrasound cohorts:

1. **Primary Development & Internal Audit Cohort (Curated ThyroidXL)**:
   - $N = 3,115$ pathology-confirmed examinations ($1,905$ benign, $1,210$ malignant) from the National Hospital of Endocrinology (Hitachi Aloka Arietta V70).
   - Partitioned strictly at the patient level into a training cohort ($N = 2,492$; $1,524$ benign, $968$ malignant) and an internal test cohort ($N = 623$; $381$ benign, $242$ malignant).
   - Zero patient overlap across splits.
2. **External Validation & Caliper Stress Cohort (DDTI)**:
   - $N = 347$ sonograms from $299$ patients (Hospital de San José / Columbia University cohort).
   - Features routine clinical sonographer measurement calipers and radiologist ACR TI-RADS risk stratifications (TR1–TR5).
   - Evaluated using both image-level analysis ($N = 347$, patient-clustered bootstrap) and patient-level aggregation ($N = 299$, mean pooling).
3. **Multi-Vendor Cross-Center Generalization Cohort (TN5000)**:
   - $N = 5,000$ images acquired on GE Logiq E9 and Logiq S7 scanners (Cancer Hospital CAMS).

---

## 📁 Repository Structure

```
.
├── README.md                      # Comprehensive project documentation
├── AGENTS.md                      # Research governance and reproducibility criteria
├── requirements.txt               # Python package dependencies
├── run_audit.sh                   # Automated end-to-end audit execution script
├── .gitignore                     # Repository hygiene filter
├── assets/
│   └── figures/                   # High-resolution clinical workflow and result figures
│       ├── figure1.jpg            # Study workflow and audit architecture
│       ├── figure2.jpg            # Multi-condition ROC discrimination curves
│       ├── figure3.jpg            # Grad-CAM attribution and caliper hijacking visualization
│       └── figure4.jpg            # Decision Curve Analysis (DCA) Net Benefit profiles
├── code/                          # Audit algorithms, models, and evaluation routines
│   ├── dataset_manifest.py        # Dataset mapping, verification, and manifest integrity checks
│   ├── mask_and_perturb.py        # ROI extraction, parenchymal inpainting, and caliper synthesis
│   ├── train_audit_models.py      # Multi-backbone training, ablation conditions, and stress tests
│   ├── evaluate_metrics.py        # AUROC, ECE, DCA net benefit, bootstrap CI, and caliper energy
│   └── run_pilot_audit.py         # Fast integration and verification smoke test
├── analysis/                      # Empirical logs, experiment summaries, and DCA outputs
│   └── experiment_results.json    # Tabular evaluation records and metric outputs
└── docs/                          # Detailed clinical protocols and study specifications
    ├── RESEARCH_BRIEF.md          # Scientific endpoints and clinical design
    ├── PROBLEM_ANCHOR.md          # Shortcut taxonomy and clinical hazard formulation
    ├── DATASET_AUDIT.md           # Cohort curation and leakage prevention protocols
    └── EXPERIMENT_PROTOCOL.md    # Multi-condition ablation and DCA validation protocol
```

---

## 🚀 Installation & Quick Start

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Hajimi-Sudo/thyroid-ultrasound-shortcut-audit.git
cd thyroid-ultrasound-shortcut-audit
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity (Smoke Test)

Run the pilot test to verify PyTorch CUDA acceleration and perturbation transformations:

```bash
python3 code/run_pilot_audit.py
```

### 3. Run the Full Multi-Condition Audit

To train and evaluate the models across all representation conditions:

```bash
# Baseline ResNet-50 audit
python3 code/train_audit_models.py --backbone resnet50 --condition all --epochs 8 --output_dir analysis

# ConvNeXt-Tiny audit
python3 code/train_audit_models.py --backbone convnext_tiny --condition all --epochs 8 --output_dir analysis

# Swin-Transformer Tiny audit
python3 code/train_audit_models.py --backbone swin_t --condition all --epochs 8 --output_dir analysis
```

Or execute the complete automated audit pipeline via shell script:

```bash
bash run_audit.sh
```

---

## 📈 Key Metric Implementations (`code/evaluate_metrics.py`)

- **Diagnostic Accuracy & Calibration**:
  - `compute_diagnostic_metrics(y_true, y_prob, threshold)`: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, Brier score, and Expected Calibration Error (ECE, 10 equal-width bins).
- **Clinical Decision Utility**:
  - `compute_decision_curve(y_true, y_prob, thresholds)`: Calculates Net Benefit across clinical FNA decision thresholds ($p_t \in [0.05, 0.50]$) for model-guided triaging vs. Treat-All vs. Treat-None.
- **Attribution Faithfulness & Caliper Energy**:
  - `compute_caliper_attribution_energy(saliency_map, caliper_mask)`: Calculates $E_{\text{caliper}}$, the fraction of Grad-CAM gradient energy situated within the dilated caliper mask.
  - `compute_pointing_game_hit_rate(saliency_maps, masks)`: Evaluates whether peak attribution falls within true nodule boundaries.
- **Safety Threshold Calibration**:
  - `calibrate_locked_high_sensitivity_threshold(y_true, y_prob, target_sensitivity=0.950)`: Determines the locked operating cutoff $p^*$ maximizing specificity while guaranteeing $\text{Sensitivity} \ge 95.0\%$.
- **Intra-Patient Clustered Bootstrap**:
  - `compute_patient_clustered_bootstrap_ci(patient_ids, y_true, y_prob, n_bootstraps=2000)`: Evaluates robust $95\%$ confidence intervals accounting for multiple images per patient in external validation cohorts.

---

## ⚖️ License & Ethical Declarations

- **Code License**: Distributed under the [MIT License](LICENSE).
- **Clinical Ethics**: All analyses were conducted on de-identified, publicly accessible retrospective ultrasound cohorts in full compliance with institutional review board standards, the Declaration of Helsinki, and HIPAA privacy rules.
