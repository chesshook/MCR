# MCR mathematical-native trainer — paste this entire file into one Kaggle cell.
# Notebook Settings: Accelerator = GPU T4 x2; Internet = On.
from pathlib import Path
import os
import subprocess
import sys

REPO = "https://github.com/chesshook/MCR.git"
BRANCH = "math-native-v1"
ROOT = Path("/kaggle/working/MCR")
if ROOT.exists() and (ROOT / ".git").exists():
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "origin", BRANCH], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(ROOT)], check=True)

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    actual = torch.cuda.device_count() if torch.cuda.is_available() else 0
    raise RuntimeError(f"Select Kaggle GPU T4 x2. Detected {actual} CUDA devices.")
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | {p.total_memory / 1024**3:.1f} GiB")
    if "T4" not in p.name.upper():
        raise RuntimeError(f"Expected a T4 at GPU {i}; found {p.name}")
print("The system is trained from random initialization. No pretrained LLM or tokenizer is downloaded.")

subprocess.run([
    sys.executable, str(ROOT / "launch_kaggle.py"),
    "--data-dir", "/kaggle/working/MCR_DATA",
    "--output-dir", "/kaggle/working/MCR_OUTPUT",
    "--chat-limit", "120000",
    "--nli-limit", "200000",
    "--epochs", "4",
    "--batch-size", "32",
    "--max-bytes", "512",
    "--resume",
], cwd=ROOT, check=True, env=os.environ.copy())
