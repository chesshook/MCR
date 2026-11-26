# MCR — copy this entire cell into a Kaggle Notebook and run it once.
# Notebook Settings: Accelerator = GPU T4 x2; Internet = On.
from pathlib import Path
import os
import subprocess
import sys

REPO = "https://github.com/chesshook/MCR.git"
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MCR"

if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=True)

# Preserve Kaggle's preinstalled torch/CUDA stack; install only the project dependencies.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "-r", str(REPO_DIR / "requirements-kaggle.txt"),
    ],
    check=True,
)

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
    raise RuntimeError(
        f"MCR requires Kaggle GPU T4 x2; detected {detected}. "
        "Choose Settings > Accelerator > GPU T4 x2 and enable Internet."
    )

print("MCR will use two synchronized data-parallel workers:")
for index in range(2):
    gpu = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {gpu.name} | {gpu.total_memory / 1024**3:.1f} GiB VRAM")
    if "T4" not in gpu.name.upper():
        raise RuntimeError(f"Expected T4 at GPU {index}, got {gpu.name}")

os.environ.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_DEBUG", "WARN")

subprocess.run(
    [
        sys.executable, str(REPO_DIR / "launch_kaggle.py"),
        "--output-dir", "/kaggle/working/MCR_OUTPUT",
        "--resume",
    ],
    cwd=REPO_DIR,
    env=os.environ.copy(),
    check=True,
)
print(
    "Finished. Save a Kaggle notebook version so MCR_OUTPUT and MCR_ADAPTER.zip "
    "are preserved as downloadable output artifacts."
)
