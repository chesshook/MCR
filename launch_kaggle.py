"""One-command training orchestration for two synchronized Kaggle T4 GPUs."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def ensure_data_dependency():
    if importlib.util.find_spec("datasets") is None:
        print("Installing the dataset reader only (no model/tokenizer libraries)...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "datasets==3.6.0"], check=True)
    import datasets  # noqa: F401
    import numpy  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/kaggle/working/MCR_DATA")
    p.add_argument("--output-dir", default="/kaggle/working/MCR_OUTPUT")
    p.add_argument("--chat-limit", type=int, default=120_000)
    p.add_argument("--nli-limit", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-bytes", type=int, default=384)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        raise RuntimeError(f"Expected Kaggle T4 x2; found {count} CUDA devices. Set Accelerator=GPU T4 x2.")
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)
        if "T4" not in props.name.upper():
            raise RuntimeError(f"GPU {index} is not a T4: {props.name}")
    print(f"CPU cores visible: {os.cpu_count()}", flush=True)
    print("Both T4s will train from scratch with DDP/NCCL gradient all-reduce.", flush=True)
    print("No pretrained model, tokenizer, embeddings, or text-generation weights are used.", flush=True)

    ensure_data_dependency()
    root = Path(__file__).resolve().parent
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "data_manifest.json"
    if manifest.exists() and all((data_dir / name).exists() for name in
                                 ("chat_train.jsonl", "chat_validation.jsonl", "nli_train.jsonl", "nli_validation.jsonl")):
        print(f"Reusing prepared data in {data_dir}", flush=True)
    else:
        cmd = [sys.executable, str(root / "prepare_data.py"), "--output-dir", str(data_dir),
               "--chat-limit", str(args.chat_limit), "--nli-limit", str(args.nli_limit)]
        subprocess.run(cmd, cwd=root, check=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                 "--nproc_per_node=2", str(root / "mcr" / "train.py"), "--data-dir", str(data_dir),
                 "--output-dir", str(output_dir), "--epochs", str(args.epochs),
                 "--batch-size", str(args.batch_size), "--max-bytes", str(args.max_bytes)]
    if args.resume or (output_dir / "training_state.pt").exists():
        train_cmd.append("--resume")
    print("\n=== Starting MCR from-scratch 2xT4 training ===", flush=True)
    subprocess.run(train_cmd, cwd=root, env=env, check=True)

    export = Path("/kaggle/working/MCR_MATH_NATIVE.zip")
    include = ["mcr_model.pt", "mcr_config.json", "mcr_training_report.json",
               "response_index.npy", "response_bank.jsonl"]
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for name in include:
            path = output_dir / name
            if not path.exists():
                raise RuntimeError(f"Training artifact missing: {path}")
            archive.write(path, arcname=name)
    print(f"\nFinished. Model export: {export}", flush=True)
    print("Save a Kaggle Notebook version to preserve MCR_OUTPUT and MCR_MATH_NATIVE.zip.", flush=True)


if __name__ == "__main__":
    main()
