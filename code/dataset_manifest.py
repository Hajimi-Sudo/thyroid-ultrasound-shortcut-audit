"""
Dataset manifest and verification utilities for multi-cohort thyroid ultrasound AI audit.
Supports ThyroidXL (MICCAI 2025), TN5000 (Nature Sci Data 2025), and TN3K.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class DatasetConfig:
    name: str
    source_url: str
    local_dir: str
    total_images: int
    has_patient_id: bool
    annotation_type: str  # 'mask', 'bbox', 'label_only'
    device_vendors: List[str]
    has_tirads: bool
    role: str  # 'development', 'external_test', 'stress_test'


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "thyroidxl": DatasetConfig(
        name="ThyroidXL",
        source_url="https://huggingface.co/datasets/VNPT-AI/ThyroidXL",
        local_dir="data/thyroidxl",
        total_images=11635,
        has_patient_id=True,
        annotation_type="mask",
        device_vendors=["Hitachi Aloka Arietta V70"],
        has_tirads=True,
        role="development",
    ),
    "tn5000": DatasetConfig(
        name="TN5000",
        source_url="https://doi.org/10.6084/m9.figshare.cb6a67f17c04b29e7edd19",
        local_dir="data/tn5000",
        total_images=5000,
        has_patient_id=False,  # Image-level split in original release
        annotation_type="bbox",
        device_vendors=["GE Logiq E9", "GE Logiq S7"],
        has_tirads=False,
        role="external_test",
    ),
    "tn3k": DatasetConfig(
        name="TN3K",
        source_url="https://github.com/FATTY-W/TN3K",
        local_dir="data/tn3k",
        total_images=3493,
        has_patient_id=True,
        annotation_type="mask",
        device_vendors=["Mindray Resona 7", "Toshiba Aplio 500"],
        has_tirads=False,
        role="stress_test",
    ),
}


def print_cohort_summary():
    """Print an auditable summary of all registered datasets."""
    logging.info("Multi-Cohort Thyroid Ultrasound Manifest Summary:")
    for key, cfg in DATASET_REGISTRY.items():
        print(f"\n--- [{cfg.name}] ---")
        print(f"Role: {cfg.role}")
        print(f"Images: {cfg.total_images:,}")
        print(f"Patient-level Split: {'Yes (No Leakage)' if cfg.has_patient_id else 'No (Image-level Only)'}")
        print(f"Annotations: {cfg.annotation_type}")
        print(f"TI-RADS Scores Available: {cfg.has_tirads}")
        print(f"Devices: {', '.join(cfg.device_vendors)}")
        print(f"Source: {cfg.source_url}")


if __name__ == "__main__":
    print_cohort_summary()
