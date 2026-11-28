# Run this entire file in ONE Kaggle notebook cell.
# Kaggle settings: GPU = T4 x2, Internet = On.
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO = "https://github.com/chesshook/MCR.git"
BRANCH = "mosaic-chatbot-v1"
ROOT = Path("/kaggle/working/MCR_MOSAIC_SOURCE")
OUTPUT = Path("/kaggle/working/MCR_MOSAIC_OUTPUT")
BUNDLE = Path("/kaggle/working/MCR_MOSAIC_CPU")
ZIP_PATH = Path("/kaggle/working/MCR_MOSAIC_CPU.zip")

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
for device_id in range(2):
    device_name = torch.cuda.get_device_name(device_id)
    print(f"GPU {device_id}: {device_name}")
    if "T4" not in device_name.upper():
        raise RuntimeError(f"Expected a Tesla T4 at GPU {device_id}, found {device_name}")

packages_to_install = []
if importlib.util.find_spec("datasets") is None:
    packages_to_install.append("datasets>=3.6,<5")
if importlib.util.find_spec("wn") is None:
    packages_to_install.append("wn==1.1.1")
if packages_to_install:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages_to_install],
        check=True,
    )

OUTPUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
env["TOKENIZERS_PARALLELISM"] = "false"

print("\n=== MOSAIC OFFLINE BAKING: TWO T4 GPUS ===", flush=True)
subprocess.run(
    [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2",
        str(ROOT / "mosaic_train.py"),
        "--output-dir", str(OUTPUT),
        "--epochs", "15",
        "--batch-size", "128",
        "--learning-rate", "0.025",
    ],
    cwd=ROOT,
    env=env,
    check=True,
)

print("\n=== EXPORTING OPEN ENGLISH WORDNET 2025+ ===", flush=True)
subprocess.run(
    [sys.executable, str(ROOT / "mosaic_export_wordnet.py"),
     "--output-dir", str(OUTPUT), "--max-senses", "4"],
    cwd=ROOT,
    env=env,
    check=True,
)

print("\n=== BUILDING NATIVE C++17 RUNTIME ===", flush=True)
BUILD = ROOT / "mosaic_core" / "build"
subprocess.run(
    ["cmake", "-S", str(ROOT / "mosaic_core"), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release"],
    check=True,
)
subprocess.run(["cmake", "--build", str(BUILD), "--parallel", "2"], check=True)
subprocess.run(["ctest", "--test-dir", str(BUILD), "--output-on-failure"], check=True)

binary_candidates = [
    BUILD / "mosaic_chat",
    BUILD / "Release" / "mosaic_chat.exe",
    BUILD / "mosaic_chat.exe",
]
binary = next((candidate for candidate in binary_candidates if candidate.exists()), None)
if binary is None:
    raise FileNotFoundError("CMake completed but mosaic_chat executable was not found.")

bench_candidates = [
    BUILD / "mosaic_bench",
    BUILD / "Release" / "mosaic_bench.exe",
    BUILD / "mosaic_bench.exe",
]
benchmark = next((candidate for candidate in bench_candidates if candidate.exists()), None)
if benchmark is None:
    raise FileNotFoundError("CMake completed but mosaic_bench executable was not found.")

if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
BUNDLE.mkdir(parents=True)
shutil.copytree(ROOT / "mosaic_core", BUNDLE / "mosaic_core",
                ignore=shutil.ignore_patterns("build", "*.o", "*.obj"))
for name in [
    "MOSAIC_CHATBOT_README.md", "mosaic_features.py",
    "mosaic_train.py", "mosaic_notebook_chat.py",
    "mosaic_export_wordnet.py", "build_windows.bat", "run_windows.bat", "LICENSE",
]:
    shutil.copy2(ROOT / name, BUNDLE / name)
shutil.copy2(OUTPUT / "intent_model.mcr", BUNDLE / "intent_model.mcr")
shutil.copy2(OUTPUT / "intent_report.json", BUNDLE / "intent_report.json")
shutil.copy2(OUTPUT / "wordnet.tsv", BUNDLE / "wordnet.tsv")
shutil.copy2(OUTPUT / "wordnet_manifest.json", BUNDLE / "wordnet_manifest.json")
shutil.copy2(binary, BUNDLE / binary.name)
shutil.copy2(benchmark, BUNDLE / benchmark.name)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file in BUNDLE.rglob("*"):
        if file.is_file():
            archive.write(file, file.relative_to(BUNDLE))

print("\n=== BAKING RESULT ===")
print((OUTPUT / "intent_report.json").read_text(encoding="utf-8"))
print(f"Quantized intent model: {(OUTPUT / 'intent_model.mcr').stat().st_size / 1024**2:.2f} MiB")
print(f"Native executable: {binary}")
print(f"Native benchmark: {benchmark}")
print(f"Package: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MiB)")
print("Inference runs in native C++; the two GPUs are not needed after this point.")

# Display a real input box in the notebook, connected to the compiled C++ process.
sys.path.insert(0, str(ROOT))
from mosaic_notebook_chat import display_chat

display_chat(
    binary=str(binary),
    model=str(BUNDLE / "intent_model.mcr"),
    memory="/kaggle/working/MOSAIC_MEMORY.tsv",
    lexicon=str(BUNDLE / "wordnet.tsv"),
)
