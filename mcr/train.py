"""Train MCR's original byte-semantic network from random initialization on T4 x2."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from mcr.encoding import encode_bytes, pad_sequences
from mcr.model import MCRModel, contrastive_loss, parameter_count
from mcr.schedule import make_task_schedule


class JsonlDataset(Dataset):
    def __init__(self, path: str, task: str, max_examples: int = 0):
        self.task = task
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if max_examples and len(self.rows) >= max_examples:
                        break
        if not self.rows:
            raise RuntimeError(f"No examples found in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(task: str, max_bytes: int):
    def collate(rows):
        import torch
        if task == "chat":
            left = [encode_bytes(row["context"], max_bytes) for row in rows]
            right = [encode_bytes(row["reply"], max_bytes) for row in rows]
            li, lm, _ = pad_sequences(left)
            ri, rm, _ = pad_sequences(right)
            return {
                "context_ids": torch.tensor(li, dtype=torch.long),
                "context_mask": torch.tensor(lm, dtype=torch.bool),
                "reply_ids": torch.tensor(ri, dtype=torch.long),
                "reply_mask": torch.tensor(rm, dtype=torch.bool),
            }
        if task == "nli":
            left = [encode_bytes(row["premise"], max_bytes) for row in rows]
            right = [encode_bytes(row["hypothesis"], max_bytes) for row in rows]
            li, lm, _ = pad_sequences(left)
            ri, rm, _ = pad_sequences(right)
            return {
                "premise_ids": torch.tensor(li, dtype=torch.long),
                "premise_mask": torch.tensor(lm, dtype=torch.bool),
                "hypothesis_ids": torch.tensor(ri, dtype=torch.long),
                "hypothesis_mask": torch.tensor(rm, dtype=torch.bool),
                "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
            }
        raise ValueError(task)
    return collate


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _encode_texts(model: MCRModel, texts: list[str], device: torch.device,
                  max_bytes: int, batch_size: int = 256) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = [encode_bytes(text, max_bytes) for text in batch]
            ids, masks, _ = pad_sequences(encoded)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            masks_t = torch.tensor(masks, dtype=torch.bool, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                vectors = model.encoder(ids_t, masks_t)
            output.append(vectors.float().cpu().numpy())
    if not output:
        return np.zeros((0, model.encoder.output_dim), dtype=np.float32)
    return np.concatenate(output, axis=0)


def evaluate_nli(model: MCRModel, rows: list[dict], device: torch.device,
                 max_bytes: int, batch_size: int = 128) -> dict:
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            collated = make_collator("nli", max_bytes)(batch)
            for key, value in collated.items():
                collated[key] = value.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(task="nli", **{k: v for k, v in collated.items() if k != "labels"})
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == collated["labels"]).sum().item())
            total += len(batch)
    return {"accuracy": correct / max(1, total), "examples": total}


def evaluate_retrieval(model: MCRModel, rows: list[dict], device: torch.device,
                       max_bytes: int, batch_size: int = 128) -> dict:
    if not rows:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "examples": 0}
    rows = rows[: min(2500, len(rows))]
    queries = _encode_texts(model, [row["context"] for row in rows], device, max_bytes, batch_size)
    replies = _encode_texts(model, [row["reply"] for row in rows], device, max_bytes, batch_size)
    scores = queries @ replies.T
    order = np.argsort(-scores, axis=1)[:, :min(5, len(rows))]
    target = np.arange(len(rows))[:, None]
    hits = (order == target)
    return {
        "recall_at_1": float(hits[:, :1].any(axis=1).mean()),
        "recall_at_5": float(hits.any(axis=1).mean()),
        "examples": len(rows),
    }


def build_response_index(model: MCRModel, rows: list[dict], output_dir: Path,
                         device: torch.device, max_bytes: int) -> None:
    # Contrastive training aligns context vectors with reply vectors. At inference,
    # a live context must query this paired-reply index (not a context-vs-context index).
    replies = [row["reply"] for row in rows]
    embeddings = _encode_texts(model, replies, device, max_bytes, batch_size=256)
    np.save(output_dir / "response_index.npy", embeddings.astype(np.float16))
    with (output_dir / "response_bank.jsonl").open("w", encoding="utf-8") as handle:
        for reply in replies:
            handle.write(json.dumps({"reply": reply}, ensure_ascii=False) + "\n")
    print(f"Built retrieval memory: {len(replies):,} context/reply entries", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32, help="Per GPU batch size")
    p.add_argument("--max-bytes", type=int, default=384)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("MCR training requires exactly two visible CUDA GPUs (Kaggle T4 x2).")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(f"Expected torchrun WORLD_SIZE=2; got {world_size}.")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    if "T4" not in torch.cuda.get_device_name(local_rank).upper():
        raise RuntimeError(f"Expected Tesla T4; found {torch.cuda.get_device_name(local_rank)}")
    print(f"[rank {rank}] {props.name}; {props.total_memory / 1024**3:.1f} GiB; device=cuda:{local_rank}", flush=True)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chat_train = JsonlDataset(str(data_dir / "chat_train.jsonl"), "chat")
    chat_val = JsonlDataset(str(data_dir / "chat_validation.jsonl"), "chat")
    nli_train = JsonlDataset(str(data_dir / "nli_train.jsonl"), "nli")
    nli_val = JsonlDataset(str(data_dir / "nli_validation.jsonl"), "nli")

    chat_sampler = DistributedSampler(chat_train, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed, drop_last=True)
    nli_sampler = DistributedSampler(nli_train, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed + 1, drop_last=True)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 1, "pin_memory": True,
                     "persistent_workers": True, "drop_last": True}
    chat_loader = DataLoader(chat_train, sampler=chat_sampler,
                             collate_fn=make_collator("chat", args.max_bytes), **loader_kwargs)
    nli_loader = DataLoader(nli_train, sampler=nli_sampler,
                            collate_fn=make_collator("nli", args.max_bytes), **loader_kwargs)
    if len(chat_loader) == 0 or len(nli_loader) == 0:
        raise RuntimeError("At least one training dataset is smaller than one distributed batch.")

    model = MCRModel().to(device)
    total_parameters = parameter_count(model)
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank,
              find_unused_parameters=True, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    steps_per_epoch = len(chat_loader) + len(nli_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(100, int(total_steps * 0.04))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-3, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    start_task_index = 0
    global_step = 0
    checkpoint_path = output_dir / "training_state.pt"
    if args.resume and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path)
        ddp.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        start_task_index = int(state.get("next_task_index", 0))
        global_step = int(state.get("global_step", 0))
        if rank == 0:
            print(f"Resuming at epoch {start_epoch + 1}, task {start_task_index}, global step {global_step}", flush=True)

    if rank == 0:
        print("\n=== MCR scratch-trained architecture ===", flush=True)
        print(f"Parameters: {total_parameters:,}", flush=True)
        print(f"Train examples: {len(chat_train):,} conversation pairs; {len(nli_train):,} NLI pairs", flush=True)
        print(f"Per-GPU batch: {args.batch_size}; global batch: {args.batch_size * world_size}", flush=True)
        print(f"Updates per epoch: {steps_per_epoch:,}; epochs: {args.epochs}; planned updates: {total_steps:,}", flush=True)
        print("All parameters start from random initialization; DDP/NCCL all-reduces gradients across both T4s.", flush=True)

    start_time = time.time()
    history_losses = []
    for epoch in range(start_epoch, args.epochs):
        chat_sampler.set_epoch(epoch)
        nli_sampler.set_epoch(epoch)
        chat_iter = iter(chat_loader)
        nli_iter = iter(nli_loader)
        tasks = make_task_schedule(len(chat_loader), len(nli_loader), args.seed + epoch)
        skip = start_task_index if epoch == start_epoch else 0
        for skipped_task in tasks[:skip]:
            if skipped_task == "chat":
                next(chat_iter)
            else:
                next(nli_iter)
        ddp.train()
        for task_index in range(skip, len(tasks)):
            task = tasks[task_index]
            batch = next(chat_iter) if task == "chat" else next(nli_iter)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if task == "chat":
                    query, reply = ddp(task="chat", **batch)
                    loss = contrastive_loss(query.float(), reply.float())
                else:
                    labels = batch.pop("labels")
                    logits = ddp(task="nli", **batch)
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            loss_value = float(loss.detach().float().item())
            history_losses.append({"task": task, "loss": loss_value})

            if rank == 0 and (global_step % 25 == 0 or global_step == 1):
                recent = history_losses[-25:]
                mean_loss = sum(x["loss"] for x in recent) / len(recent)
                elapsed = time.time() - start_time
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"task={task} loss25={mean_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={elapsed / 60:.1f}m", flush=True,
                )
            if global_step % args.save_steps == 0:
                if rank == 0:
                    temp_path = checkpoint_path.with_suffix(".tmp")
                    torch.save({
                        "model": ddp.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "next_task_index": task_index + 1,
                        "global_step": global_step,
                        "config": {"max_bytes": args.max_bytes, "output_dim": 1024},
                    }, temp_path)
                    temp_path.replace(checkpoint_path)
                torch.distributed.barrier()
        # Carry clean state to the next epoch and make resume deterministic.
        start_task_index = 0
        if rank == 0:
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "model": ddp.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "next_task_index": 0,
                "global_step": global_step,
                "config": {"max_bytes": args.max_bytes, "output_dim": 1024},
            }, temp_path)
            temp_path.replace(checkpoint_path)
        torch.distributed.barrier()

    if rank == 0:
        plain_model = ddp.module
        plain_model.eval()
        torch.save(plain_model.state_dict(), output_dir / "mcr_model.pt")
        config = {
            "architecture": "MCR ByteSemanticEncoder + NLI relation head",
            "parameters": total_parameters,
            "input": "UTF-8 bytes; zero is padding; ids 1..256 are byte values + 1",
            "max_bytes": args.max_bytes,
            "output_dim": 1024,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "autoregressive_decoder": False,
            "epochs": args.epochs,
            "per_gpu_batch_size": args.batch_size,
            "world_size": world_size,
            "optimizer": "AdamW",
            "contrastive_temperature": 0.07,
        }
        (output_dir / "mcr_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("\n=== Held-out evaluation ===", flush=True)
        nli_metrics = evaluate_nli(plain_model, nli_val.rows, device, args.max_bytes)
        retrieval_metrics = evaluate_retrieval(plain_model, chat_val.rows, device, args.max_bytes)
        print("NLI:", json.dumps(nli_metrics), flush=True)
        print("Retrieval:", json.dumps(retrieval_metrics), flush=True)
        build_response_index(plain_model, chat_train.rows, output_dir, device, args.max_bytes)
        report = {
            "architecture": config,
            "dataset_manifest": json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8")),
            "global_optimizer_steps": global_step,
            "elapsed_minutes": round((time.time() - start_time) / 60, 2),
            "heldout_nli": nli_metrics,
            "heldout_response_retrieval": retrieval_metrics,
            "honest_scope": (
                "This is a from-scratch, non-autoregressive semantic encoder trained for "
                "sentence-relation classification and retrieval-based conversational replies. "
                "It is not a general-purpose free-form text generator or proof of human-level understanding."
            ),
        }
        (output_dir / "mcr_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n=== MCR training report ===", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
