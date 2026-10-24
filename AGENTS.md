# Thyroid Ultrasound AI Shortcut Learning & Generalizability Audit

## Working title

Beyond Diagnostic Accuracy: Clinical Audit of Shortcut Learning and Non-Lesion Confounders in Deep Learning Models for Thyroid Nodule Ultrasound Analysis

## Scope

- Discipline: Medicine / Thyroid Endocrinology / Ultrasound Imaging / Clinical AI Safety
- Study type: Multi-cohort retrospective diagnostic and algorithmic safety audit
- Intended target: Clinical Sonography & Medical Imaging AI Journal (Under Peer Review)
- Primary clinical endpoint: Robustness of nodule malignancy classification across imaging conditions and centers, reduction of false-positive predictions and unnecessary fine-needle aspiration (FNA) biopsy risk
- Clinical reference: ACR TI-RADS (American College of Radiology Thyroid Imaging Reporting and Data System) guidelines and Bethesda cytopathology categories
- Datasets:
  - Development and internal test: **ThyroidXL** (11,635 images, 4,093 patients; patient-level split; pathology-confirmed; expert segmentation; ACR TI-RADS annotated; Hitachi Aloka Arietta V70)
  - External generalizability test: **TN5000** (5,000 images, GE Logiq E9/S7; multi-machine cancer center cohort; bounding box annotated)
  - Caliper artifact stress test: **TN3K / DDTI** (public cohorts with radiologist measurement caliper markers)

## Frozen boundaries

- Do not frame the manuscript as a pure computer science / computer vision algorithm paper. The narrative must remain firmly anchored in clinical endocrinology, diagnostic safety, avoidance of overdiagnosis/unnecessary biopsies, and adherence to TI-RADS guidelines.
- Do not use the official TN5000 random train/test split as an internal benchmark, because its split was performed at the image level, presenting a fatal patient-level data leakage risk. Use TN5000 strictly as an external validation cohort for models trained on ThyroidXL.
- Strictly distinguish between **technical spurious shortcuts** (measurement calipers, vendor UI, text, depth bars, sector beam boundaries) and **biological parenchymal background features** (diffuse background hypoechogenicity/heterogeneity indicative of concurrent Hashimoto's thyroiditis).
- Do not use simple sharp black/gray box masking that introduces high-frequency edge artifacts (out-of-distribution step functions) to occlude nodules. Use smooth Gaussian blending, texture-matching, or generative inpainting.
- Do not rely solely on visual saliency maps (Grad-CAM) to claim interpretability. Quantify attribution with localization metrics (Energy-inside-mask ratio / Pointing Game) and perturbation benchmarks (ROAR/KAR).
- Report clinical decision utility metrics: Sensitivity, Specificity, Positive/Negative Predictive Values (PPV/NPV under clinical prevalence), Expected Calibration Error (ECE), and Decision Curve Analysis (DCA net benefit).

## Fatal flaws to avoid

1. Patient-level data leakage between training and validation sets.
2. Claiming background prediction is purely an algorithmic shortcut when it reflects biological thyroid parenchymal disease (e.g., Hashimoto's thyroiditis).
3. Conflating internal high AUC on clean data with clinical readiness, while ignoring cross-device collapse.
4. Omitting comparison with established clinical scoring guidelines (ACR TI-RADS).
5. Failing to provide decision curve analysis (DCA) or unnecessary biopsy reduction estimates.

## Local LaTeX environment

Always compile with the local TeX Live 2022 binaries per workspace instructions:
`PATH=/root/code/rh-workspace/latex/texlive/2022/bin/x86_64-linux:$PATH`
Helper script: `/root/code/rh-workspace/latex/use-texlive2022.sh`
