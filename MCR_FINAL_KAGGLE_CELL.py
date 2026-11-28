# SINGLE KAGGLE CELL — copy everything in this cell and click Run.
# Kaggle settings: Accelerator = GPU T4 x2, Internet = On.
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/chesshook/MCR.git"
BRANCH = "mosaic-natural-chat-v1"
ROOT = Path("/kaggle/working/MCR_NATURAL_SOURCE")

if ROOT.exists() and (ROOT / ".git").exists():
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "origin", BRANCH], check=True)
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(ROOT)],
        check=True,
    )

CELL = ROOT / "MOSAIC_NATURAL_CHAT_ONE_CELL.py"
if not CELL.is_file():
    raise FileNotFoundError(f"Training cell not found: {CELL}")

print("Launching the complete MCR training, export, native build and chat workflow.", flush=True)
exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), {"__name__": "__main__"})
