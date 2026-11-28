# MCR CPU-Native V2 — single Kaggle cell
# Uses Kaggle CPU only. The exported inference executable is native C++17.
# No CUDA, PyTorch, pretrained model, tokenizer, or LLM is downloaded.

from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

REPO = "https://github.com/chesshook/MCR.git"
BRANCH = "cpu-native-v2"
ROOT = Path("/kaggle/working/MCR_CPU_NATIVE_V2_SRC")
CPU = ROOT / "cpu_native"
OUT = Path("/kaggle/working/MCR_CPU_NATIVE_V2_OUTPUT")
OUT.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None, capture=False):
    print("\n$", " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        check=True,
        text=True,
        env=os.environ.copy(),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        print(result.stdout)
    return result

if (ROOT / ".git").exists():
    run(["git", "-C", ROOT, "fetch", "origin", BRANCH])
    run(["git", "-C", ROOT, "checkout", BRANCH])
    run(["git", "-C", ROOT, "pull", "--ff-only", "origin", BRANCH])
else:
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, ROOT])

print("\n=== 1. Download labelled intent dataset ===")
run([sys.executable, "prepare_clinc.py"], cwd=CPU)

print("\n=== 2. Train compact sparse intent model on CPU ===")
MODEL = OUT / "mcr_intent_model.txt"
run([
    sys.executable, "train_intent.py",
    "--train", "data/clinc150/train.tsv",
    "--valid", "data/clinc150/valid.tsv",
    "--output", MODEL,
    "--epochs", "12",
    "--target-selective-accuracy", "0.90",
], cwd=CPU)

print("\n=== 3. Compile C++17 inference runtime ===")
EXE = OUT / ("mcr_intent.exe" if os.name == "nt" else "mcr_intent")
run([
    "g++", "-std=c++17", "-O3", "-DNDEBUG",
    str(CPU / "src" / "main.cpp"), "-o", EXE
])

print("\n=== 4. Evaluate untouched test split ===")
evaluation = run([
    sys.executable, "evaluate_intent.py",
    "--model", MODEL,
    "--test", "data/clinc150/test.tsv",
], cwd=CPU, capture=True)
(OUT / "test_metrics.txt").write_text(evaluation.stdout, encoding="utf-8")

print("\n=== 5. Native-runtime smoke test ===")
smoke = subprocess.run(
    [str(EXE), str(MODEL)],
    input="Hi\nThanks for your help\nGoodbye\nTell me a joke about quantum physics\n",
    text=True, capture_output=True, check=True,
)
print(smoke.stderr)
print(smoke.stdout)
(OUT / "smoke_test.txt").write_text(smoke.stdout, encoding="utf-8")

print("\n=== 6. Export portable source + trained model ===")
archive = Path("/kaggle/working/MCR_CPU_NATIVE_V2.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in CPU.rglob("*"):
        if path.is_file() and "data/clinc150/" not in path.as_posix() and "__pycache__" not in path.as_posix():
            zf.write(path, Path("cpu_native") / path.relative_to(CPU))
    zf.write(MODEL, "artifacts/mcr_intent_model.txt")
    zf.write(OUT / "test_metrics.txt", "artifacts/test_metrics.txt")
    zf.write(OUT / "smoke_test.txt", "artifacts/smoke_test.txt")
    zf.write(EXE, "artifacts/" + EXE.name)

print("\nBUILD COMPLETE")
print("Model:", MODEL)
print("Native executable:", EXE)
print("Portable package:", archive)
print("This release is an intent-classification component, not yet a conversational chatbot.")
