# Research Brief: Clinical Audit of Shortcut Learning in Thyroid Ultrasound AI

## Executive Summary

- **Working Title**: Beyond Diagnostic Accuracy: Clinical Audit of Shortcut Learning and Non-Lesion Confounders in Deep Learning Models for Thyroid Nodule Ultrasound Analysis
- **Target Journal**: Specialized Medical AI & Clinical Sonography Journal (Under Peer Review)
- **Target Section**: *Thyroid Endocrinology* (Alternative: *Cancer Endocrinology*)
- **Article Type**: Original Research (Type A)
- **Clinical Relevance**: In thyroid endocrinology, overdiagnosis of indolent thyroid cancers (e.g., small papillary thyroid microcarcinomas) and high rates of unnecessary fine-needle aspiration (FNA) biopsies represent significant healthcare burdens. While AI tools are being developed to triage nodules, hidden reliance on technical artifacts (calipers, machine frames, scanning protocols) rather than genuine morphologic criteria introduces diagnostic instability and unpredictable clinical risks.

## Core Research Questions

1. **Quantification of Non-Lesion Shortcut Dependency**:
   - To what extent can standard deep learning classifiers predict nodule malignancy when the nodule itself is completely occluded and only non-lesion background tissue is provided?
   - How much diagnostic performance degrades when vendor frames, depth markers, and peripheral text are stripped?

2. **Caliper-Induced False Alarm Vulnerability**:
   - When sonographers measure suspicious nodules, they leave caliper crosses (cross-hairs/dashed lines) on the screen. Does deep learning models treat these human-added measurement marks as an authoritative shortcut for malignancy?
   - How does introducing synthetic calipers onto benign nodules alter the model's false-positive rate and predicted risk probabilities?

3. **Multi-Center Generalization Collapse**:
   - How does a model trained on a clean single-vendor dataset (ThyroidXL, Hitachi Aloka) perform when evaluated on an independent, multi-machine hospital cohort (TN5000, GE Logiq)?
   - Which input formulation (full image vs. cleaned full image vs. cropped nodule ROI) exhibits the smallest cross-center performance drop?

4. **Clinical Decision Utility vs. ACR TI-RADS**:
   - How does shortcut mitigation (ROI-focused learning) affect decision curve analysis (DCA) and the net reduction of unnecessary biopsies across low-to-intermediate risk nodules (TI-RADS 3 and 4)?

## Target Journal Profile: Specialized Medical AI & Clinical Sonography Journal (Under Peer Review)

- **Audience**: Clinical endocrinologists, endocrine surgeons, oncologists, thyroid sonographers.
- **Tone & Style**: Patient-centric, biologically grounded, clinically actionable. Avoid pure CS jargon unless explicitly defined in clinical terms.
- **Reporting Standards**: Follow STARD (Standards for Reporting Diagnostic Accuracy) and CLAIM (Checklist for Artificial Intelligence in Medical Imaging) guidelines.
- **Key Required Comparisons**:
  - Direct comparison with ACR TI-RADS guidelines.
  - Subgroup analysis by nodule size (microcarcinomas $\le 10\text{ mm}$ vs. larger nodules $> 10\text{ mm}$).
  - Stratification by background thyroid echogenicity (normal parenchyma vs. heterogeneous/hypoechoic background of Hashimoto's thyroiditis).
  - Decision Curve Analysis (DCA) demonstrating net clinical benefit in the 10%–30% threshold probability window (the standard FNA decision boundary).

## Success Criteria

1. Demonstrating statistically significant diagnostic drop when models are tested on external data without ROI constraint.
2. Demonstrating that background-only images achieve higher-than-chance AUC, confirming non-lesion shortcut exploitation in standard unconstrained models.
3. Proving that nodule-cropped or de-biased models maintain clinical sensitivity ($\ge 95\%$) while significantly improving specificity and net clinical benefit on external validation cohorts.
4. Production of a complete, compilable LaTeX manuscript adhering to the standard medical Vancouver reference style.
