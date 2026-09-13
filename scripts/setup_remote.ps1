# One-time setup on the Windows PC with the RTX 3070. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_remote.ps1
# Needs Python 3.11/3.12 and an NVIDIA driver recent enough for CUDA 12.6 (nvidia-smi shows "CUDA Version: 12.6" or higher).

$ErrorActionPreference = "Stop"
nvidia-smi
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('GPU:', torch.cuda.get_device_name(0))"
python -m pytest tests -q
Write-Host "`nReady. Next:  python src/run_all.py --seed 0   (then: python src/run_all.py)"
