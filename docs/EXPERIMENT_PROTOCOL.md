# Experiment Protocol: Multi-Condition Ablation & Generalizability Audit

## 1. Experimental Intervention Conditions

To systematically isolate and quantify shortcut learning, five distinct input conditions are audited:

1. **Condition 1 (Raw Full-Frame Baseline)**:
   - Complete uncropped ultrasound scan as exported by the ultrasound machine, including vendor UI banners, depth scale bars, sector acoustic boundaries, and peripheral text ($X^{(1)} = X \in \mathbb{R}^{H \times W \times C}$).
   - Evaluates the unconstrained baseline network $M_{\text{full}}$ trained on $2,492$ cases and tested on $623$ cases.

2. **Condition 2 (Cleaned Full-Frame)**:
   - Ultrasound scan cropped to the acoustic fan region ($5\%$ lateral/inferior, $8\%$ superior) to strip peripheral equipment banners and time stamps while preserving entire acoustic background ($X^{(2)} = \mathcal{T}_{\text{clean}}(X)$).
   - Evaluates the frozen baseline network $M_{\text{full}}$ under test-time border cropping.

3. **Condition 3 (Inpainted Background / Lesion Occluded)**:
   - The central nodule region is masked and filled with surrounding parenchymal texture using fast marching telea inpainting ($X^{(3)} = \mathcal{T}_{\text{inpaint}}(X, M)$).
   - Evaluates the frozen baseline network $M_{\text{full}}$. Quantifies pure non-lesion shortcut capacity: any performance above random chance directly demonstrates background information leakage.

4. **Condition 4 (Caliper-Stressed Counterfactual Perturbation)**:
   - Standard 3-pixel-wide electronic measurement crosshairs ($+$) matching clinical sonography dimensions are counterfactually injected across all $623$ test images (both $358$ benign and $265$ malignant) ($X^{(4)} = \mathcal{T}_{\text{caliper}}(X)$).
   - Calipers are strictly excluded during training to prevent artificial training-set associations.
   - Evaluates paired within-nodule probability shifts ($\Delta \hat{y}_i = \hat{y}_i^{(\text{caliper})} - \hat{y}_i^{(\text{clean})}$), paired $t$-test, Wilcoxon signed-rank test, sensitivity collapse, and Grad-CAM caliper attribution energy $E_{\text{caliper}}$.

5. **Condition 5 (Nodule ROI-Constrained Safeguard)**:
   - Bounding box cropped tightly around the nodule with a $10\%$ anatomical margin to preserve nodule-parenchyma margin boundaries ($X^{(5)} = \mathcal{T}_{\text{roi}}(X, B, 0.10)$).
   - A dedicated safeguard model $M_{\text{roi}}$ is trained on $2,492$ nodule ROIs and tested on $623$ test ROIs, forcing feature representation strictly on genuine nodular morphology.

---

## 2. Model Architectures & Training Standards

To ensure findings are architecture-agnostic, three paradigms are benchmarked:
- **ResNet-50**: Deep residual convolutional network.
- **ConvNeXt-Tiny**: Pure convolutional architecture with large $7\times 7$ kernels.
- **Swin-Transformer (Swin-T)**: Hierarchical vision transformer utilizing shifted local window self-attention.

### Training Hyperparameters
- **Optimizer**: AdamW with weight decay $1\times 10^{-4}$
- **Learning Rate Schedule**: Cosine annealing, initial LR $1\times 10^{-4}$, warmup 5 epochs, total 8–50 epochs.
- **Batch Size**: 32
- **Data Augmentation**: Random horizontal flip, subtle rotation ($\pm 10^\circ$), brightness/contrast jitter ($\pm 0.1$). No geometric distortion altering aspect ratio.
- **Loss Function**: Binary Cross-Entropy with logits.

---

## 3. Evaluation Metrics Suite

### 1. Diagnostic Discrimination & Calibration
- **AUROC**: Area under ROC curve with 95% DeLong confidence intervals.
- **AUPRC**: Area under precision-recall curve.
- **Sensitivity & Specificity**: At default threshold ($0.50$) and locked safety threshold ($p^*$).
- **Expected Calibration Error (ECE)**: 10 equal-width reliability bins.
- **Brier Score**: Mean squared probabilistic deviation.

### 2. Clinical Utility (Decision Curve Analysis)
- **Net Benefit ($NB$)**:
  $$NB(p_t) = \frac{TP}{N} - \frac{FP}{N} \left(\frac{p_t}{1 - p_t}\right)$$
  evaluated across clinical fine-needle aspiration (FNA) decision thresholds ($p_t \in [0.05, 0.50]$).
- **Benign Biopsy Avoidance Rate**: Proportion of confirmed benign lesions safely classified below the locked high-sensitivity safety cutoff ($p^*$ targeting $\ge 95.0\%$ sensitivity).

### 3. Attribution Faithfulness & Saliency Energy
- **Caliper Attribution Energy Fraction ($E_{\text{caliper}}$)**:
  $$E_{\text{caliper}} = \frac{\sum_{(x,y) \in M_{\text{caliper}}} A(x,y)}{\sum_{(x,y) \in \Omega} A(x,y)}$$
  where $A(x,y)$ is the Grad-CAM attribution intensity and $M_{\text{caliper}}$ is the 3-pixel dilated caliper mask.
- **Pointing Game Hit Rate**: Proportion of test examinations where $\arg\max A(x,y) \in \Omega_{\text{nodule}}$.

### 4. External Validation & Cluster-Aware Statistics
- **Patient-Clustered Bootstrap**: Resampling 2,000 bootstrap iterations clustered by patient identifier to compute valid 95% confidence intervals on the DDTI external cohort ($N = 347$ images, $299$ patients).
- **Patient-Level Aggregation**: Evaluating patient-level triage safety via mean and maximum risk probability pooling.
