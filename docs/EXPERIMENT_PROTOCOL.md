# Experiment Protocol: Multi-Condition Ablation & Generalizability Audit

## 1. Experimental Intervention Conditions

To systematically isolate and quantify shortcut learning, five distinct input conditions are created for each image:

1. **Condition 1 (Raw Full-Frame)**:
   - Complete uncropped ultrasound scan as saved by the ultrasound machine, including vendor UI borders, depth scale bars, sector acoustic boundaries, and peripheral text.
   - Purpose: Baseline representative of unconstrained "end-to-end" clinical AI pipelines.

2. **Condition 2 (Cleaned Full-Frame)**:
   - Ultrasound scan cropped to the acoustic fan region only, stripping all peripheral vendor metadata, time stamps, and depth bars.
   - Purpose: Isolates peripheral machine metadata shortcuts from thyroid tissue.

3. **Condition 3 (Nodule ROI Only)**:
   - Tight bounding box around the nodule with a standard 10% anatomical margin to preserve marginal boundary definition.
   - Purpose: Forces the network to learn exclusively from the nodule's intrinsic morphologic characteristics (composition, echogenicity, margin, shape, microcalcifications).

4. **Condition 4 (Inpainted Background / Lesion Occluded)**:
   - The nodule region (defined by the ground-truth segmentation mask or bounding box) is removed and filled using texture-preserving inpainting based on surrounding non-nodular thyroid parenchyma.
   - Purpose: The definitive shortcut test. Any diagnostic accuracy above random chance (AUC 0.50) directly quantifies non-lesion shortcut exploitation.

5. **Condition 5 (Synthetic Caliper Stress-Test)**:
   - Benign nodules are randomly stamped with synthetic cross-hair measurement markers (calipers) matching clinical sonography dimensions.
   - Purpose: Quantifies whether the model shifts its prediction toward malignancy upon seeing caliper markers.

---

## 2. Model Architectures & Training Standards

To ensure findings are not idiosyncratic to a single architecture, three standard backbones are benchmarked:
- **ResNet-50**: Standard convolutional baseline with residual inductive bias.
- **ConvNeXt-Tiny**: Modernized pure convolutional architecture with large kernels ($7\times 7$).
- **Swin-Transformer (Swin-T)**: Hierarchical vision transformer utilizing shifted local window self-attention.

### Training Hyperparameters
- **Optimizer**: AdamW with weight decay $1\times 10^{-4}$
- **Learning Rate Schedule**: Cosine annealing, initial LR $1\times 10^{-4}$, warmup 5 epochs, total 50 epochs.
- **Batch Size**: 32
- **Data Augmentation**: Random horizontal flip, random rotation ($\pm 10^\circ$), color jitter (brightness $\pm 0.1$, contrast $\pm 0.1$). No aggressive geometric distortion that distorts nodule aspect ratios (taller-than-wide).
- **Loss Function**: Binary Cross-Entropy with label smoothing (0.05) or Focal Loss ($\gamma = 2$).

---

## 3. Evaluation Metrics Suite

### 1. Diagnostic Discrimination
- **AUROC (Area Under ROC Curve)**: With 95% DeLong confidence intervals.
- **AUPRC (Area Under Precision-Recall Curve)**: Critical for class-imbalanced settings.
- **Sensitivity at High Specificity**: Sensitivity fixed at 95% specificity, reflecting the clinical priority of not over-calling benign nodules.
- **Unnecessary Biopsy Avoidance Rate**: Proportion of benign nodules correctly classified as low-risk (not requiring FNA).

### 2. Calibration & Reliability
- **Expected Calibration Error (ECE)**: 10 equal-width bins.
- **Brier Score**: Quadratic penalty reflecting probabilistic accuracy.

### 3. Clinical Utility (Decision Curve Analysis)
- **Net Benefit ($NB$)**:
  $$NB(p_t) = \frac{TP}{N} - \frac{FP}{N} \left(\frac{p_t}{1 - p_t}\right)$$
  where $p_t \in [0.05, 0.40]$ represents the clinician's decision threshold for recommending FNA.

### 4. Attribution & Saliency Faithfulness
- **Energy-Inside-Mask Ratio ($E_{in}$)**:
  $$E_{in} = \frac{\sum_{(x,y) \in \text{Mask}} A(x,y)}{\sum_{(x,y)} A(x,y)}$$
  where $A(x,y)$ is the Grad-CAM attribution intensity.
- **Pointing Game Hit Rate**: Proportion of test images where the maximum attribution point $\arg\max A(x,y)$ falls inside the ground-truth nodule mask.
