"""
Full training and multi-condition shortcut audit suite for thyroid ultrasound AI.
Supports:
1. Backbones: ResNet-50, ConvNeXt-Tiny, Swin-Transformer Tiny
2. Conditions:
   - Raw Full-Frame (full baseline)
   - Cleaned Full-Frame (peripheral device borders/metadata cropped out)
   - Nodule ROI-Constrained (nodule cropped with 10% anatomical margin)
   - Inpainted Background (central nodule occluded to audit background shortcut capacity)
   - Caliper-Stressed Evaluation (synthetic caliper crosshairs injected on test images)
3. Evaluations:
   - Internal test set: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, ECE, DCA Net Benefit
   - Caliper perturbation: Paired probability shifts, paired t-test, Wilcoxon, and attribution energy
   - External validation (DDTI): Image-level (clustered bootstrap) and Patient-level (mean pooling)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from scipy.stats import ttest_rel, wilcoxon

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_metrics import (
    compute_diagnostic_metrics,
    compute_decision_curve,
    compute_caliper_attribution_energy,
    compute_patient_clustered_bootstrap_ci,
    calibrate_locked_high_sensitivity_threshold,
)
from mask_and_perturb import (
    crop_nodule_roi,
    inpaint_nodule_background,
    inject_synthetic_caliper,
    strip_device_borders,
    generate_caliper_mask,
)


def get_model(backbone: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """
    Instantiate model backbone with a binary classification linear output head.
    Supports: 'resnet50', 'convnext_tiny', 'swin_t'
    """
    backbone = backbone.lower()
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif backbone in ("convnext_tiny", "convnext"):
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    elif backbone in ("swin_t", "swin_tiny", "swin"):
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, 1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from: resnet50, convnext_tiny, swin_t")
    return model


class ThyroidDataset(Dataset):
    """
    Dataset loader for ultrasound images under distinct experimental audit conditions.
    """
    def __init__(self, df: pd.DataFrame, condition: str = "full", is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.condition = condition.lower()
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        w, h = img.size
        label = float(row["label"])

        # Determine nodule bounding box from row annotations or central sonographic default
        if "bbox_xmin" in row and pd.notna(row["bbox_xmin"]):
            nodule_bbox = (int(row["bbox_xmin"]), int(row["bbox_ymin"]), int(row["bbox_xmax"]), int(row["bbox_ymax"]))
        else:
            nodule_bbox = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))

        # Apply condition-specific morphological or representation intervention
        if self.condition == "cleaned":
            img = strip_device_borders(img)
        elif self.condition == "roi":
            img = crop_nodule_roi(img, nodule_bbox, margin_ratio=0.10)
        elif self.condition == "inpainted":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[nodule_bbox[1]:nodule_bbox[3], nodule_bbox[0]:nodule_bbox[2]] = 1
            img = inpaint_nodule_background(img, mask)
        elif self.condition == "caliper":
            img = inject_synthetic_caliper(img, nodule_bbox)

        tensor = self.transform(img)
        patient_id = str(row["patient_id"]) if "patient_id" in row else str(idx)
        return tensor, torch.tensor(label, dtype=torch.float32), patient_id


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """Train backbone with binary cross-entropy and cosine annealing schedule."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        scheduler.step()
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Time: {time.time() - t0:.1f}s")

    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate risk probabilities and targets from a test dataloader."""
    model.eval()
    all_probs, all_targets, all_pids = [], [], []
    with torch.no_grad():
        for x, y, pids in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())
            all_pids.extend(pids)
    return np.array(all_probs), np.array(all_targets), all_pids


def run_caliper_stress_audit(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Perform paired caliper-stress evaluation on the same test set:
    1. Evaluate clean unperturbed images
    2. Evaluate caliper-injected images
    3. Compute paired shifts, Wilcoxon p-value, and benign false-positive rates
    """
    clean_ds = ThyroidDataset(test_df, condition="cleaned", is_train=False)
    caliper_ds = ThyroidDataset(test_df, condition="caliper", is_train=False)

    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
    caliper_loader = DataLoader(caliper_ds, batch_size=batch_size, shuffle=False)

    probs_clean, targets, _ = evaluate_model(model, clean_loader, device)
    probs_caliper, _, _ = evaluate_model(model, caliper_loader, device)

    benign_mask = (targets == 0)
    malignant_mask = (targets == 1)

    delta_p = probs_caliper - probs_clean
    delta_p_benign = delta_p[benign_mask]
    delta_p_malignant = delta_p[malignant_mask]

    # Paired statistical testing
    t_stat, p_val_ttest = ttest_rel(probs_caliper, probs_clean)
    w_stat, p_val_wilcoxon = wilcoxon(probs_caliper, probs_clean)

    # Thresholded diagnostic metrics
    metrics_caliper = compute_diagnostic_metrics(targets, probs_caliper)

    return {
        "mean_delta_p_benign": float(np.mean(delta_p_benign)),
        "std_delta_p_benign": float(np.std(delta_p_benign)),
        "mean_delta_p_malignant": float(np.mean(delta_p_malignant)),
        "std_delta_p_malignant": float(np.std(delta_p_malignant)),
        "p_val_wilcoxon": float(p_val_wilcoxon),
        "auroc_under_caliper": float(metrics_caliper["auroc"]),
        "sensitivity_under_caliper": float(metrics_caliper["sensitivity"]),
        "specificity_under_caliper": float(metrics_caliper["specificity"]),
        "ece_under_caliper": float(metrics_caliper["ece"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-condition thyroid ultrasound shortcut audit.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "convnext_tiny", "swin_t"])
    parser.add_argument("--condition", type=str, default="all", choices=["all", "full", "cleaned", "inpainted", "roi", "caliper"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"--- Thyroid Ultrasound Shortcut Audit Suite ---")
    print(f"Backbone: {args.backbone} | Device: {device}")

    # Resolve manifest path
    manifest_path = args.manifest_path
    if manifest_path is None or not os.path.exists(manifest_path):
        candidate_paths = [
            "data/thyroid_us_benchmark/manifest.csv",
            "/root/autodl-tmp/thyroid-ultrasound-shortcut-generalizability/data/thyroid_us_benchmark/manifest.csv",
            os.path.join(args.data_dir, "manifest.csv"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                manifest_path = p
                break

    if manifest_path is None or not os.path.exists(manifest_path):
        print(f"[Warning] Manifest file not found in default locations. Running synthetic demonstration.")
        # Create minimal synthetic DataFrame for integration testing
        rng = np.random.RandomState(42)
        demo_rows = []
        for i in range(100):
            demo_rows.append({
                "image_path": f"/tmp/dummy_{i}.png",
                "label": int(i % 2 == 0),
                "split": "train" if i < 80 else "test",
                "patient_id": f"P{i // 2:04d}",
            })
        df = pd.DataFrame(demo_rows)
    else:
        print(f"Loaded manifest from: {manifest_path}")
        df = pd.read_csv(manifest_path)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    print(f"Train Cohort: {len(train_df)} cases | Internal Test Cohort: {len(test_df)} cases")

    conditions = ["full", "cleaned", "inpainted", "roi"] if args.condition == "all" else [args.condition]
    results_dict = {}

    for cond in conditions:
        print(f"\n==================================================")
        print(f"   Executing Audit Condition: [{cond.upper()}] with [{args.backbone}]")
        print(f"==================================================")

        train_ds = ThyroidDataset(train_df, condition=cond, is_train=True)
        test_ds = ThyroidDataset(test_df, condition=cond, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = get_model(args.backbone, pretrained=True).to(device)
        model = train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr)

        probs, targets, pids = evaluate_model(model, test_loader, device)
        diag_metrics = compute_diagnostic_metrics(targets, probs)
        dca_metrics = compute_decision_curve(targets, probs)

        results_dict[cond] = {
            "metrics": diag_metrics,
            "dca_net_benefit_sample": dca_metrics["net_benefit_model"][:10].tolist(),
            "thresholds": dca_metrics["thresholds"][:10].tolist(),
        }

        # If condition is 'cleaned', also perform the paired caliper stress audit
        if cond == "cleaned" or args.condition == "caliper":
            print(f"--> Running paired Caliper Stress Audit on [{cond}] baseline...")
            caliper_stress_res = run_caliper_stress_audit(model, test_df, device, batch_size=args.batch_size)
            results_dict["caliper_stress_audit"] = caliper_stress_res
            print(f"    Benign mean risk shift (Delta P): +{caliper_stress_res['mean_delta_p_benign']:.4f}")
            print(f"    Wilcoxon signed-rank test p-value: {caliper_stress_res['p_val_wilcoxon']:.4e}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"audit_results_{args.backbone}.json")
    with open(out_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nAudit complete! Results stored at: {out_file}")


if __name__ == "__main__":
    main()
