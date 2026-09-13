#!/usr/bin/env bash
# One-time setup on a Linux machine with the RTX 3070 (or any CUDA GPU). Run from the repo root:
#   bash scripts/setup_remote.sh
set -euo pipefail
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('GPU:', torch.cuda.get_device_name(0))"
python -m pytest tests -q
echo
echo "Ready. Next:  python src/run_all.py --seed 0   (then: python src/run_all.py)"
