"""Train MOSAIC's compact intent recognizer from scratch on CLINC150.

Two-process NCCL DDP is used on Kaggle T4 x2. The exported model is an int8
hashed n-gram linear classifier consumed by the native C++ runtime.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from mosaic_features import FEATURE_DIM, extract_features


class SparseIntentNet(nn.Module):
    def __init__(self, feature_dim: int, classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            feature_dim, classes, mode="sum", sparse=False,
            include_last_offset=False
        )
        self.bias = nn.Parameter(torch.zeros(classes))

    def forward(self, ids, offsets, values, lengths):
        logits = self.embedding(ids, offsets, per_sample_weights=values)
        logits = logits / torch.sqrt(torch.clamp(lengths, min=1.0)).unsqueeze(1)
        return logits + self.bias


def encode_split(rows, text_key, label_key):
    encoded = []
    labels = []
    for row in rows:
        encoded.append(extract_features(str(row[text_key])))
        labels.append(int(row[label_key]))
    return encoded, labels


def make_batch(encoded, labels, indices, device):
    ids, values, offsets, lengths, targets = [], [], [], [], []
    cursor = 0
    for index in indices:
        features = encoded[int(index)]
        offsets.append(cursor)
        lengths.append(max(1, len(features)))
        for feature_id, feature_value in features:
            ids.append(feature_id)
            values.append(feature_value)
            cursor += 1
        targets.append(labels[int(index)])
    if not ids:
        raise RuntimeError("empty training batch")
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(lengths, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(model, encoded, labels, device, batch_size=512):
    model.eval()
    predictions = []
    for start in range(0, len(labels), batch_size):
        indices = list(range(start, min(start + batch_size, len(labels))))
        batch = make_batch(encoded, labels, indices, device)
        logits = model(batch[0], batch[1], batch[2], batch[3])
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    correct = sum(int(a == b) for a, b in zip(predictions, labels))
    classes = max(labels) + 1
    tp = [0] * classes
    fp = [0] * classes
    fn = [0] * classes
    for expected, predicted in zip(labels, predictions):
        if expected == predicted:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1
    f1_values = []
    for i in range(classes):
        denom = 2 * tp[i] + fp[i] + fn[i]
        f1_values.append((2 * tp[i] / denom) if denom else 0.0)
    return {
        "accuracy": correct / max(1, len(labels)),
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "examples": len(labels),
    }


def save_int8_model(path: Path, model, class_names):
    weight = model.embedding.weight.detach().float().cpu().contiguous()
    bias = model.bias.detach().float().cpu().numpy().astype("<f4", copy=False)
    peak = float(weight.abs().max().item())
    scale = peak / 127.0 if peak > 0 else 1.0
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    quantized_np = quantized.contiguous().numpy()

    with path.open("wb") as output:
        output.write(struct.pack(
            "<8sIIf", b"MCRINT1\0", weight.shape[0], weight.shape[1], scale
        ))
        for label in class_names:
            raw = str(label).encode("utf-8")
            if not raw or len(raw) > 1024:
                raise ValueError(f"invalid class label: {label!r}")
            output.write(struct.pack("<H", len(raw)))
            output.write(raw)
        output.write(bias.tobytes(order="C"))
        output.write(quantized_np.tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/kaggle/working/MCR_MOSAIC_OUTPUT")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world > 1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Kaggle baking run.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.manual_seed(20261128)
    np.random.seed(20261128)

    if rank == 0:
        print("Loading CLINC150 plus (150 intents + out-of-scope class)...", flush=True)
    dataset = load_dataset("clinc/clinc_oos", "plus")
    train_split = dataset["train"]
    label_key = "intent" if "intent" in train_split.column_names else "label"
    text_key = "text"
    label_feature = train_split.features[label_key]
    class_names = list(label_feature.names)
    classes = len(class_names)
    if classes < 2 or classes > 512:
        raise RuntimeError(f"unexpected number of intent labels: {classes}")

    train_x, train_y = encode_split(train_split, text_key, label_key)
    val_x, val_y = encode_split(dataset["validation"], text_key, label_key)
    test_x, test_y = encode_split(dataset["test"], text_key, label_key)
    if rank == 0:
        print(
            f"Dataset: train={len(train_y)} validation={len(val_y)} "
            f"test={len(test_y)} classes={classes}",
            flush=True
        )

    net = SparseIntentNet(FEATURE_DIM, classes).to(device)
    model = DistributedDataParallel(net, device_ids=[local_rank]) if use_ddp else net
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    sampler = DistributedSampler(
        list(range(len(train_y))), num_replicas=world, rank=rank,
        shuffle=True, seed=42, drop_last=False
    )
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    best_val = -1.0
    best_state = None
    started = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        indices = list(iter(sampler))
        running_loss = 0.0
        batches = 0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch = make_batch(train_x, train_y, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch[0], batch[1], batch[2], batch[3])
            loss = F.cross_entropy(logits, batch[4], label_smoothing=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += float(loss.detach().item())
            batches += 1

        if use_ddp:
            dist.barrier()
        if rank == 0:
            eval_model = model.module if use_ddp else model
            validation = evaluate(eval_model, val_x, val_y, device)
            mean_loss = running_loss / max(1, batches)
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"rank0_loss={mean_loss:.4f} "
                f"validation_accuracy={validation['accuracy']:.4f} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True
            )
            if validation["accuracy"] > best_val:
                best_val = validation["accuracy"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in eval_model.state_dict().items()
                }
        if use_ddp:
            dist.barrier()

    if rank == 0:
        eval_model = model.module if use_ddp else model
        if best_state is not None:
            eval_model.load_state_dict(best_state)
        validation = evaluate(eval_model, val_x, val_y, device)
        test = evaluate(eval_model, test_x, test_y, device)
        save_int8_model(output_dir / "intent_model.mcr", eval_model, class_names)

        report = {
            "architecture": "hashed word/character n-gram EmbeddingBag linear intent model",
            "feature_dimension": FEATURE_DIM,
            "classes": classes,
            "parameters": sum(p.numel() for p in eval_model.parameters()),
            "quantization": "symmetric int8 weights, float32 bias",
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "autoregressive_decoder": False,
            "dataset": "clinc/clinc_oos:plus",
            "license": "CC BY 3.0; preserve attribution and verify upstream terms",
            "train_examples": len(train_y),
            "validation": validation,
            "heldout_test": test,
            "epochs": args.epochs,
            "world_size": world,
            "training_minutes": round((time.time() - started) / 60.0, 2),
            "scope_warning": (
                "Intent classification is not general language understanding. "
                "The C++ runtime combines this model with explicit dialogue state, "
                "fact operations and bounded deterministic skills."
            ),
        }
        (output_dir / "intent_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        torch.save(
            {"model": eval_model.state_dict(), "report": report},
            output_dir / "intent_training_checkpoint.pt"
        )
        print(json.dumps(report, indent=2), flush=True)
        print(f"MODEL_EXPORTED={output_dir / 'intent_model.mcr'}", flush=True)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
