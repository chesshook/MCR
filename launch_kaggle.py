"""One-command Kaggle T4x2 orchestration: prepare data once, then train with DDP."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/MCR_OUTPUT")
    parser.add_argument("--data-dir", default="/kaggle/working/MCR_TOKENIZED_DATA")
    parser.add_argument("--smol-limit", type=int, default=50_000)
    parser.add_argument("--ultra-limit", type=int, default=25_000)
    parser.add_argument("--oasst-limit", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        raise SystemExit(
            f"MCR requires Kaggle GPU T4 x2. CUDA devices detected: {count}. "
            "Set Notebook Settings > Accelerator > GPU T4 x2 and turn Internet on."
        )
    print("=== Kaggle hardware allocation ===", flush=True)
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB VRAM", flush=True)
        if "T4" not in props.name.upper():
            raise SystemExit(f"GPU {index} is {props.name!r}; this recipe is configured for T4.")
    print(f"CPU cores visible: {os.cpu_count()}; Kaggle provides the runtime RAM.", flush=True)
    print(
        "Both GPUs will run synchronized DDP data-parallel workers; VRAM is not pooled.",
        flush=True,
    )

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    prep_cmd = [
        sys.executable, str(root / "prepare_data.py"),
        "--output-dir", str(data_dir),
        "--smol-limit", str(args.smol_limit),
        "--ultra-limit", str(args.ultra_limit),
        "--oasst-limit", str(args.oasst_limit),
        "--max-length", str(args.max_length),
    ]
    manifest = data_dir / "mcr_data_manifest.json"
    dataset_marker = data_dir / "dataset_dict.json"
    if dataset_marker.exists() and manifest.exists():
        print(f"\n=== Reusing prepared tokenized data at {data_dir} ===", flush=True)
    else:
        print("\n=== Preparing multi-source training data once on CPU ===", flush=True)
        if data_dir.exists():
            print(f"Removing incomplete tokenized-data directory: {data_dir}", flush=True)
            shutil.rmtree(data_dir)
        subprocess.run(prep_cmd, cwd=root, env=env, check=True)

    train_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2",
        str(root / "train.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--epochs", str(args.epochs),
    ]
    if args.resume or output_dir.exists():
        train_cmd.append("--resume")
    print("\n=== Starting synchronized 2xT4 QLoRA training ===", flush=True)
    subprocess.run(train_cmd, cwd=root, env=env, check=True)

    # The post-training smoke test uses GPU 0 after DDP processes have exited.
    smoke_env = env.copy()
    smoke_env["CUDA_VISIBLE_DEVICES"] = "0"
    print("\n=== Post-training conversational smoke test ===", flush=True)
    subprocess.run(
        [
            sys.executable, str(root / "chat.py"),
            "--adapter-dir", str(output_dir),
            "--smoke-test",
        ],
        cwd=root,
        env=smoke_env,
        check=True,
    )

    export_path = output_dir.parent / "MCR_ADAPTER.zip"
    with zipfile.ZipFile(
        export_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
    print(f"\nFinal adapter export: {export_path}", flush=True)
    print(
        "MCR run completed. Save the Kaggle notebook version to preserve output files.",
        flush=True,
    )


if __name__ == "__main__":
    main()
