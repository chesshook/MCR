# MOSAIC / MCR — one-time natural-language baking.
# Kaggle Notebook settings: GPU = T4 x2, Internet = On.
# The exported chat runtime is native C++; GPUs are used only for training.

from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO = "https://github.com/chesshook/MCR.git"
BRANCH = "mosaic-natural-chat-v1"
ROOT = Path("/kaggle/working/MCR_NATURAL_SOURCE")
SOURCE = ROOT / "mosaic_natural"
OUTPUT = Path("/kaggle/working/MCR_NATURAL_OUTPUT")
PACKAGE = Path("/kaggle/working/MCR_NATURAL_CPU")
ZIP_PATH = Path("/kaggle/working/MCR_NATURAL_CPU.zip")

if ROOT.exists() and (ROOT / ".git").exists():
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "origin", BRANCH], check=True)
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(ROOT)],
        check=True,
    )

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    raise RuntimeError(f"Select Kaggle GPU T4 x2. Detected {count} CUDA devices.")
for gpu in range(2):
    name = torch.cuda.get_device_name(gpu)
    print(f"GPU {gpu}: {name}")
    if "T4" not in name.upper():
        raise RuntimeError(f"Expected Tesla T4 x2, found {name}")

if importlib.util.find_spec("datasets") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "datasets>=3.6,<5"],
        check=True,
    )
if importlib.util.find_spec("ipywidgets") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets"], check=True)

OUTPUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"

print("\n=== TRAINING MOSAIC FROM RANDOM INITIALIZATION ON T4 x2 ===", flush=True)
subprocess.run(
    [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2",
        "--module", "train",
        "--output-dir", str(OUTPUT),
        "--epochs", "3",
        "--batch-size", "24",
        "--learning-rate", "0.0018",
        "--max-train-examples", "250000",
        "--max-eval-examples", "12000",
        "--vocab-size", "8192",
        "--hidden-size", "320",
        "--layers", "2",
    ],
    cwd=SOURCE,
    env=env,
    check=True,
)

print("\n=== BUILDING NATIVE C++ DIALOGUE GENERATOR ===", flush=True)
BUILD = SOURCE / "cpp" / "build"
subprocess.run(
    ["cmake", "-S", str(SOURCE / "cpp"), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release"],
    check=True,
)
subprocess.run(["cmake", "--build", str(BUILD), "--parallel", "2"], check=True)
subprocess.run(["ctest", "--test-dir", str(BUILD), "--output-on-failure"], check=True)

binary_candidates = [
    BUILD / "mosaic_natural_chat",
    BUILD / "Release" / "mosaic_natural_chat.exe",
    BUILD / "mosaic_natural_chat.exe",
]
bench_candidates = [
    BUILD / "mosaic_natural_bench",
    BUILD / "Release" / "mosaic_natural_bench.exe",
    BUILD / "mosaic_natural_bench.exe",
]
binary = next((p for p in binary_candidates if p.exists()), None)
benchmark = next((p for p in bench_candidates if p.exists()), None)
if binary is None or benchmark is None:
    raise FileNotFoundError("C++ chat or benchmark executable was not produced.")

if PACKAGE.exists():
    shutil.rmtree(PACKAGE)
PACKAGE.mkdir(parents=True)
shutil.copytree(SOURCE, PACKAGE / "mosaic_natural",
                ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"))
shutil.copy2(OUTPUT / "mosaic_dialogue.mos", PACKAGE / "mosaic_dialogue.mos")
shutil.copy2(OUTPUT / "mosaic_dialogue_report.json", PACKAGE / "mosaic_dialogue_report.json")
shutil.copy2(binary, PACKAGE / binary.name)
shutil.copy2(benchmark, PACKAGE / benchmark.name)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file in PACKAGE.rglob("*"):
        if file.is_file():
            archive.write(file, file.relative_to(PACKAGE))

print("\n=== MODEL REPORT ===")
print((OUTPUT / "mosaic_dialogue_report.json").read_text(encoding="utf-8"))
print(f"\nINT8 dialogue model: {(OUTPUT / 'mosaic_dialogue.mos').stat().st_size / 1024**2:.2f} MiB")
print(f"Native CPU executable: {binary}")
print(f"Native benchmark: {benchmark}")
print(f"Full package: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MiB)")
print("The C++ model now generates replies locally; CUDA is not used for chat.")

# Show the actual conversation input area in this notebook.
sys.path.insert(0, str(SOURCE))
from notebook_chat import display_chat

display_chat(
    binary=binary,
    model=PACKAGE / "mosaic_dialogue.mos",
    history="/kaggle/working/MOSAIC_NATURAL_HISTORY.tsv",
)
