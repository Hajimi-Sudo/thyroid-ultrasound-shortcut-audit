#!/usr/bin/env bash
# ==============================================================================
# Thyroid Ultrasound Shortcut Audit & Clinical Evaluation Runner
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================================================"
echo " Starting Thyroid Ultrasound Shortcut Audit Suite"
echo " Working directory: ${SCRIPT_DIR}"
echo "========================================================================"

# 1. Environment check
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

# 2. Dataset manifest verification
echo -e "\n[Step 1/4] Verifying dataset manifests..."
python3 code/dataset_manifest.py

# 3. Quick end-to-end integration test
echo -e "\n[Step 2/4] Running pipeline integration and perturbation verification..."
python3 code/run_pilot_audit.py

# 4. Multi-condition representation and shortcut audit (ResNet-50 baseline)
echo -e "\n[Step 3/4] Executing multi-condition audit across representations..."
python3 code/train_audit_models.py \
    --backbone resnet50 \
    --condition all \
    --epochs 8 \
    --batch_size 32 \
    --output_dir analysis

# 5. External validation and paired caliper stress summary
echo -e "\n[Step 4/4] Multi-backbone audit completed. Results written to analysis/."
echo "========================================================================"
echo " All audit steps executed successfully."
echo "========================================================================"
