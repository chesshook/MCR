"""QLoRA fine-tuning with true two-GPU DistributedDataParallel on Kaggle T4x2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class CausalChatCollator:
    """Dynamically pad chat examples and discard metadata columns."""
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before building batches.")
        self.pad_id = tokenizer.pad_token_id
        self.multiple = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(row["input_ids"]) for row in features)
        if self.multiple:
            max_len = ((max_len + self.multiple - 1) // self.multiple) * self.multiple
        input_ids, attention_masks, labels = [], [], []
        for row in features:
            length = len(row["input_ids"])
            padding = max_len - length
            input_ids.append(list(row["input_ids"]) + [self.pad_id] * padding)
            attention_masks.append(list(row["attention_mask"]) + [0] * padding)
            labels.append(list(row["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select Kaggle GPU T4 x2 and enable Internet.")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"This run is configured for exactly 2 T4 GPUs; found {torch.cuda.device_count()}."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gpu_name = torch.cuda.get_device_name(local_rank)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"Expected an NVIDIA T4 at local rank {local_rank}; found {gpu_name!r}.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(
        f"[rank {rank}/{world_size}] device={gpu_name} "
        f"memory={torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GiB",
        flush=True,
    )
    if world_size != 2:
        raise RuntimeError(f"Expected torchrun with WORLD_SIZE=2; received {world_size}.")

    data = load_from_disk(args.data_dir)
    if "train" not in data or "validation" not in data:
        raise RuntimeError("Prepared dataset must contain train and validation splits.")
    if not len(data["train"]) or not len(data["validation"]):
        raise RuntimeError("Prepared training or validation dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"[rank {rank}] Loading {args.model_id} as 4-bit NF4 onto cuda:{local_rank}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Check architecture compatibility instead of silently training zero adapters.
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    desired = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    targets = [name for name in desired if name in module_suffixes]
    if not {"q_proj", "v_proj", "o_proj"}.issubset(set(targets)):
        raise RuntimeError(
            "Could not find expected attention projections in model. "
            f"Found module suffixes: {sorted(module_suffixes)[:80]}"
        )
    print(f"[rank {rank}] LoRA target modules: {targets}", flush=True)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    if rank == 0:
        model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        ddp_timeout=1800,
        eval_accumulation_steps=8,
        report_to=[],
        save_safetensors=True,
        seed=42,
        data_seed=42,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=CausalChatCollator(tokenizer),
        processing_class=tokenizer,
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if args.resume and output_dir.exists() else None
    if rank == 0:
        print(f"Training examples: {len(data['train']):,}; validation: {len(data['validation']):,}", flush=True)
        print(
            f"Effective global batch size: "
            f"{args.batch_size * args.gradient_accumulation * world_size}",
            flush=True,
        )
        print(
            "Parallelism: two synchronized DDP replicas; each T4 owns one full 4-bit model replica.",
            flush=True,
        )
        print("NCCL all-reduce synchronizes the trainable LoRA gradients.", flush=True)
        print(f"Resume checkpoint: {checkpoint or 'none'}", flush=True)

    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_seconds = time.time() - started
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    evaluation = trainer.evaluate()
    if trainer.is_world_process_zero():
        manifest_path = Path(args.data_dir) / "mcr_data_manifest.json"
        data_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        report = {
            "model_id": args.model_id,
            "training_seconds": round(train_seconds, 2),
            "world_size": world_size,
            "parallelism": "DDP + NCCL all-reduce gradient synchronization",
            "precision": (
                "4-bit NF4 base weights, FP16 compute, LoRA trainable parameters "
                "managed by PEFT"
            ),
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_global_batch_size": args.batch_size * args.gradient_accumulation * world_size,
                "learning_rate": args.learning_rate,
                "max_sequence_length": 2048,
            },
            "lora": {"r": 32, "alpha": 64, "dropout": 0.05, "targets": targets},
            "data_manifest": data_manifest,
            "train_metrics": train_result.metrics,
            "heldout_metrics": evaluation,
            "warning": (
                "This is a fine-tuned adapter on an open pretrained model, "
                "not a foundation model pretrained from random initialization."
            ),
        }
        report_path = output_dir / "mcr_training_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("\n=== MCR training report ===", flush=True)
        print(json.dumps(report, indent=2, default=str), flush=True)
        print(f"\nAdapter and tokenizer saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
