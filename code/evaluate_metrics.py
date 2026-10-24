"""
Diagnostic and clinical decision evaluation suite for thyroid ultrasound AI audit.
Implements:
1. Classification discrimination: AUROC (with DeLong / Clustered Bootstrap CI), AUPRC, Sensitivity, Specificity, PPV, NPV
2. Model calibration: Expected Calibration Error (ECE) and Brier Score
3. Clinical utility: Decision Curve Analysis (DCA Net Benefit)
4. Saliency attribution faithfulness: Pointing Game Hit Rate, Energy-Inside-Mask, and Caliper Attribution Energy
5. Intra-patient clustered bootstrap resampling for external validation
6. High-sensitivity (target >= 95.0%) operating threshold calibration and locking
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, brier_score_loss
from typing import Dict, Tuple, List, Optional


def compute_diagnostic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Compute standard clinical diagnostic accuracy and calibration metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # ROC and PR AUC
    auroc = roc_auc_score(y_true, y_prob)
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_pts, precision_pts)

    # Thresholded classification
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (10 bins)
    ece = compute_ece(y_true, y_prob, n_bins=10)

    # Clinical Biopsy Avoidance Rate (Proportion of benign nodules spared from FNA)
    benign_total = tn + fp
    biopsy_avoidance_rate = tn / benign_total if benign_total > 0 else 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "brier_score": float(brier),
        "ece": float(ece),
        "biopsy_avoidance_rate": float(biopsy_avoidance_rate),
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform Decision Curve Analysis (DCA) to calculate Net Benefit across clinical decision thresholds.
    Standard thyroid FNA biopsy thresholds typically lie within [0.05, 0.40].
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 46)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = np.zeros_like(thresholds)

    for pt in thresholds:
        # Model net benefit
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefit_model.append(nb)

        # Treat-all net benefit (FNA biopsy for all detected nodules)
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        net_benefit_all.append(nb_all)

    return {
        "thresholds": thresholds,
        "net_benefit_model": np.array(net_benefit_model),
        "net_benefit_all": np.array(net_benefit_all),
        "net_benefit_none": net_benefit_none,
    }


def compute_caliper_attribution_energy(
    saliency_map: np.ndarray,
    caliper_mask: np.ndarray,
) -> float:
    """
    Compute caliper attribution energy fraction E_caliper.
    E_caliper = sum_{caliper} L^c(i,j) / sum_{canvas} L^c(i,j)
    """
    sal_pos = np.maximum(saliency_map, 0.0)
    total_energy = np.sum(sal_pos)
    if total_energy <= 1e-6:
        return 0.0
    caliper_energy = np.sum(sal_pos[caliper_mask > 0])
    return float(caliper_energy / total_energy)


def compute_pointing_game_hit_rate(
    saliency_maps: List[np.ndarray],
    masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate feature attribution faithfulness via Pointing Game.
    A 'Hit' occurs if the maximum attribution peak resides inside the ground-truth nodule mask.
    Energy-Inside-Mask measures the proportion of total attribution inside the nodule.
    """
    total = len(saliency_maps)
    hits = 0
    energy_ratios = []

    for sal, mask in zip(saliency_maps, masks):
        max_idx = np.unravel_index(np.argmax(sal), sal.shape)
        if mask[max_idx] > 0:
            hits += 1

        sal_pos = np.maximum(sal, 0)
        total_energy = np.sum(sal_pos)
        if total_energy > 1e-6:
            in_mask_energy = np.sum(sal_pos[mask > 0])
            energy_ratios.append(in_mask_energy / total_energy)
        else:
            energy_ratios.append(0.0)

    hit_rate = hits / total if total > 0 else 0.0
    mean_energy_ratio = float(np.mean(energy_ratios)) if energy_ratios else 0.0

    return {
        "pointing_game_hit_rate": float(hit_rate),
        "mean_energy_inside_mask": mean_energy_ratio,
    }


def calibrate_locked_high_sensitivity_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.950,
) -> float:
    """
    Calibrate and lock the operating probability cutoff p* maximizing specificity subject to Sensitivity >= target_sensitivity.
    Used for pre-specified safety evaluation on held-out internal test and external validation cohorts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    threshold_candidates = np.sort(np.unique(y_prob))
    best_cutoff = threshold_candidates[0]
    best_spec = -1.0

    for cutoff in threshold_candidates:
        y_pred = (y_prob >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= target_sensitivity:
            if spec > best_spec:
                best_spec = spec
                best_cutoff = cutoff

    return float(best_cutoff)


def compute_patient_clustered_bootstrap_ci(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_baseline: Optional[np.ndarray] = None,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute patient-clustered bootstrap confidence intervals for image-level metrics.
    Accounts for intra-patient correlation in cohorts where individual patients contribute multiple scans.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Pre-index patient image locations
    patient_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    boot_aurocs = []
    boot_diffs = []

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([patient_to_indices[pid] for pid in sampled_patients])

        sub_true = y_true[sampled_indices]
        sub_prob = y_prob[sampled_indices]

        # Ensure both classes present
        if len(np.unique(sub_true)) < 2:
            continue

        auc_val = roc_auc_score(sub_true, sub_prob)
        boot_aurocs.append(auc_val)

        if y_prob_baseline is not None:
            sub_base = y_prob_baseline[sampled_indices]
            auc_base = roc_auc_score(sub_true, sub_base)
            boot_diffs.append(auc_val - auc_base)

    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))

    results = {
        "auroc_mean": float(np.mean(boot_aurocs)),
        "auroc_ci_95": (ci_lower, ci_upper),
    }

    if boot_diffs:
        diff_mean = float(np.mean(boot_diffs))
        diff_se = float(np.std(boot_diffs))
        z_stat = diff_mean / diff_se if diff_se > 0 else 0.0
        # Two-sided empirical p-value
        p_val = 2.0 * min(np.mean(np.array(boot_diffs) <= 0), np.mean(np.array(boot_diffs) >= 0))
        results["diff_mean"] = diff_mean
        results["diff_se"] = diff_se
        results["diff_z"] = float(z_stat)
        results["diff_pval"] = float(p_val)

    return results
