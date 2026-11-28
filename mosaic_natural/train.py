"""Train MOSAIC's from-scratch recurrent dialogue generator on T4 x2.

No pretrained models, model weights, embeddings, or tokenizer are loaded.
Training learns a lexical segmentation table and all GRU parameters from
random initialization. The exported artifact is consumed by native C++.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data import load_conversational_corpora, make_response_examples
from tokenizer import (
    PAD_ID, BOS_ID, USER_ID, ASSISTANT_ID, TURN_ID, BYTE_BASE, PIECE_BASE,
    LearnedTokenizer,
)


SEED = 20261128
HIDDEN_SIZE = 320
NUM_LAYERS = 2
VOCAB_SIZE = 8192
MAX_CONTEXT = 352
MAX_TARGET = 128
MAX_SEQUENCE = MAX_CONTEXT + MAX_TARGET + 4


class DialogueDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_dialogue(batch):
    inputs = []
    labels = []
    for sequence, target_start in batch:
        if len(sequence) < 2:
            continue
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.full((len(sequence) - 1,), -100, dtype=torch.long)
        first_label = max(0, target_start - 1)
        y[first_label:] = torch.tensor(sequence[target_start:], dtype=torch.long)
        inputs.append(x)
        labels.append(y)

    if not inputs:
        raise RuntimeError("collate received an empty dialogue batch")
    return (
        pad_sequence(inputs, batch_first=True, padding_value=PAD_ID),
        pad_sequence(labels, batch_first=True, padding_value=-100),
    )


class MOSAICDialogueGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, layers=NUM_LAYERS):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)

        self.embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=PAD_ID
        )
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=0.10 if self.layers > 1 else 0.0,
        )
        # Tied input/output weights: one compact lexical matrix serves both.
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoded, _ = self.gru(embedded)
        logits = F.linear(encoded, self.embedding.weight, self.output_bias)
        return logits


def evaluate(model, examples, device, batch_size=16, max_batches=1200):
    model.eval()
    loader = DataLoader(
        DialogueDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dialogue,
        num_workers=0,
    )
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            if batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
                loss_sum = F.cross_entropy(
                    logits.float().reshape(-1, model.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            count = int((labels != -100).sum().item())
            total_loss += float(loss_sum.item())
            total_tokens += count
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss_per_assistant_token": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
        "assistant_tokens": total_tokens,
    }


def quantize_tensor(tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return scale, quantized


def write_quantized(output, tensor):
    scale, quantized = quantize_tensor(tensor)
    output.write(struct.pack("<f", scale))
    output.write(quantized.tobytes(order="C"))


def export_native_model(path, model, tokenizer, max_context=512):
    state = model.state_dict()
    vocab_size = model.vocab_size
    hidden = model.hidden_size
    layers = model.layers

    # Binary format is explicitly little-endian. Vocabulary entries are stored
    # as raw UTF-8/byte sequences to preserve byte-fallback behavior exactly.
    with path.open("wb") as output:
        output.write(b"MOSGRU1\0")
        output.write(struct.pack(
            "<5I", vocab_size, hidden, layers, max_context,
            tokenizer.max_piece_bytes,
        ))
        for piece in tokenizer.pieces:
            if not piece or len(piece) > 65535:
                raise ValueError("invalid tokenizer piece for model export")
            output.write(struct.pack("<H", len(piece)))
            output.write(piece)

        write_quantized(output, state["embedding.weight"])
        for layer in range(layers):
            write_quantized(output, state[f"gru.weight_ih_l{layer}"])
            write_quantized(output, state[f"gru.weight_hh_l{layer}"])
            output.write(
                state[f"gru.bias_ih_l{layer}"].detach().float().cpu().numpy()
                .astype("<f4", copy=False).tobytes(order="C")
            )
            output.write(
                state[f"gru.bias_hh_l{layer}"].detach().float().cpu().numpy()
                .astype("<f4", copy=False).tobytes(order="C")
            )
        output.write(
            state["output_bias"].detach().float().cpu().numpy()
            .astype("<f4", copy=False).tobytes(order="C")
        )


@torch.no_grad()
def sample_reply(model, tokenizer, history, device, max_new=70):
    model.eval()
    sequence = [BOS_ID]
    for role, message in history[-8:]:
        sequence.append(USER_ID if role == "user" else ASSISTANT_ID)
        sequence.extend(tokenizer.encode(message))
        sequence.append(TURN_ID)
    sequence.append(ASSISTANT_ID)

    hidden = None
    logits = None
    for token in sequence:
        embedded = model.embedding(torch.tensor([[token]], device=device))
        out, hidden = model.gru(embedded, hidden)
        logits = F.linear(out[:, -1, :], model.embedding.weight, model.output_bias)

    generated = []
    for _ in range(max_new):
        scores = logits[0].float()
        scores[PAD_ID:BOS_ID + 1] = -1e9
        scores[USER_ID] = -1e9
        scores[ASSISTANT_ID] = -1e9
        next_id = int(torch.argmax(scores).item())
        if next_id == TURN_ID:
            break
        generated.append(next_id)
        embedded = model.embedding(torch.tensor([[next_id]], device=device))
        out, hidden = model.gru(embedded, hidden)
        logits = F.linear(out[:, -1, :], model.embedding.weight, model.output_bias)
    return tokenizer.decode(generated).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/kaggle/working/MCR_NATURAL_OUTPUT")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.0018)
    parser.add_argument("--max-train-examples", type=int, default=250000)
    parser.add_argument("--max-eval-examples", type=int, default=12000)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Select Kaggle GPU T4 x2 for the one-time baking run.")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two visible GPUs; found {torch.cuda.device_count()}.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group("nccl", init_method="env://")

    started = time.time()
    print(f"[rank {rank}] Loading dialogue corpora...", flush=True)
    train_conversations, validation_conversations, test_conversations, manifest = (
        load_conversational_corpora()
    )

    if not validation_conversations:
        rng = random.Random(SEED)
        rng.shuffle(train_conversations)
        validation_count = max(100, int(len(train_conversations) * 0.02))
        validation_conversations = train_conversations[:validation_count]
        train_conversations = train_conversations[validation_count:]
        manifest["fallback_validation_split"] = "2% dialogue-level split from training corpus"

    random.Random(SEED).shuffle(train_conversations)
    random.Random(SEED + 1).shuffle(validation_conversations)
    random.Random(SEED + 2).shuffle(test_conversations)
    print(
        f"[rank {rank}] conversations: train={len(train_conversations)} "
        f"validation={len(validation_conversations)} test={len(test_conversations)}",
        flush=True,
    )

    if rank == 0:
        tokenizer = LearnedTokenizer.train(
            (text for conversation in train_conversations for _, text in conversation),
            vocab_size=args.vocab_size,
        )
    else:
        tokenizer = None

    # Tokenizer is small enough to broadcast through the process group.
    payload = [tokenizer.pieces if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if rank != 0:
        tokenizer = LearnedTokenizer(payload[0], max_piece_bytes=32)

    print(
        f"[rank {rank}] learned vocabulary={tokenizer.vocab_size} "
        f"from training conversations only",
        flush=True,
    )

    train_examples = make_response_examples(
        train_conversations, tokenizer,
        max_context=MAX_CONTEXT, max_target=MAX_TARGET,
        max_examples=args.max_train_examples,
    )
    val_examples = make_response_examples(
        validation_conversations, tokenizer,
        max_context=MAX_CONTEXT, max_target=MAX_TARGET,
        max_examples=args.max_eval_examples,
    )
    test_examples = make_response_examples(
        test_conversations, tokenizer,
        max_context=MAX_CONTEXT, max_target=MAX_TARGET,
        max_examples=args.max_eval_examples,
    )

    rng = random.Random(SEED)
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    rng.shuffle(test_examples)
    train_examples = train_examples[:args.max_train_examples]
    val_examples = val_examples[:args.max_eval_examples]
    test_examples = test_examples[:args.max_eval_examples]

    if len(train_examples) < 1000:
        raise RuntimeError(
            f"Only {len(train_examples)} assistant training examples loaded. "
            "Refusing to call this a conversational training run."
        )
    if not val_examples:
        raise RuntimeError("No held-out validation examples were produced.")
    if not test_examples:
        raise RuntimeError("No held-out test examples were produced.")

    print(
        f"[rank {rank}] response examples: train={len(train_examples)} "
        f"validation={len(val_examples)} test={len(test_examples)}",
        flush=True,
    )

    train_dataset = DialogueDataset(train_examples)
    sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=rank,
        shuffle=True, seed=SEED, drop_last=True,
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_dialogue, num_workers=0, pin_memory=True,
        drop_last=True,
    )

    model = MOSAICDialogueGRU(
        tokenizer.vocab_size, args.hidden_size, args.layers
    ).to(device)
    ddp = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        ddp.parameters(), lr=args.learning_rate, weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.08
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    checkpoint_path = output_dir / "mosaic_dialogue_checkpoint.pt"
    best_validation = float("inf")
    best_state = None
    global_step = 0
    start_epoch = 0

    if args.resume and checkpoint_path.exists():
        print(f"[rank {rank}] Resuming checkpoint: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ddp.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        best_state = checkpoint["best_state"]
        if start_epoch > args.epochs:
            raise RuntimeError("Saved checkpoint is from a later epoch count than this run.")
        print(
            f"[rank {rank}] Resume epoch={start_epoch + 1}, "
            f"global_step={global_step}, best_validation={best_validation:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        ddp.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_tokens = 0
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                logits = ddp(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, tokenizer.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            valid_tokens = int((labels != -100).sum().item())
            epoch_loss += float(loss.detach().item()) * valid_tokens
            epoch_tokens += valid_tokens
            global_step += 1

            if rank == 0 and (global_step % 100 == 0 or batch_index == 0):
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={float(loss.detach().item()):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    f"elapsed={(time.time() - started) / 60:.1f}m",
                    flush=True,
                )

        dist.barrier()
        if rank == 0:
            validation = evaluate(
                ddp.module, val_examples, device, batch_size=args.batch_size
            )
            training_loss = epoch_loss / max(epoch_tokens, 1)
            print(
                f"EPOCH_RESULT epoch={epoch + 1} "
                f"train_token_loss={training_loss:.4f} "
                f"validation_loss={validation['loss_per_assistant_token']:.4f} "
                f"validation_ppl={validation['perplexity']:.2f} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
            if validation["loss_per_assistant_token"] < best_validation:
                best_validation = validation["loss_per_assistant_token"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in ddp.module.state_dict().items()
                }

            # Save atomically after every completed epoch. A restarted Kaggle
            # session resumes here rather than losing every earlier epoch.
            checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": ddp.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_validation": best_validation,
                    "best_state": best_state,
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, checkpoint_path)
            print(
                f"CHECKPOINT_SAVED next_epoch={epoch + 2} "
                f"global_step={global_step}",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        if best_state is not None:
            ddp.module.load_state_dict(best_state)

        validation = evaluate(
            ddp.module, val_examples, device, batch_size=args.batch_size
        )
        test = evaluate(
            ddp.module, test_examples, device, batch_size=args.batch_size
        )

        samples = []
        for conversation in test_conversations[:5]:
            history = conversation[-3:-1] if len(conversation) >= 2 else conversation
            if history and history[-1][0] == "user":
                samples.append({
                    "history": history,
                    "generated_reply_greedy": sample_reply(
                        ddp.module, tokenizer, history, device
                    ),
                })

        model_path = output_dir / "mosaic_dialogue.mos"
        export_native_model(model_path, ddp.module, tokenizer, max_context=MAX_CONTEXT)
        report = {
            "architecture": "MOSAIC from-scratch 2-layer recurrent dialogue language model",
            "generator": "GRU causal dialogue model with tied input/output embeddings",
            "tokenizer": "training-corpus lexical/subword vocabulary with UTF-8 byte fallback",
            "vocabulary_size": tokenizer.vocab_size,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "parameters": sum(p.numel() for p in ddp.module.parameters()),
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_llm": False,
            "autoregressive_generation": True,
            "training_objective": "assistant-turn next-token cross entropy conditioned on dialogue history",
            "sources": manifest,
            "conversation_counts": {
                "train": len(train_conversations),
                "validation": len(validation_conversations),
                "test": len(test_conversations),
            },
            "response_example_counts": {
                "train": len(train_examples),
                "validation": len(val_examples),
                "test": len(test_examples),
            },
            "validation": validation,
            "heldout_test": test,
            "epochs": args.epochs,
            "resumed_from_epoch": start_epoch,
            "global_optimizer_steps": global_step,
            "world_size": world,
            "per_gpu_batch_size": args.batch_size,
            "training_minutes": round((time.time() - started) / 60, 2),
            "export_model_bytes": model_path.stat().st_size,
            "generated_samples": samples,
            "scope": (
                "The model is trained for open-ended conversational language "
                "continuation and reply generation. It is not claimed to have "
                "human-level intelligence or reliable external factual knowledge."
            ),
        }
        (output_dir / "mosaic_dialogue_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        torch.save(
            {"model": ddp.module.state_dict(), "report": report},
            output_dir / "mosaic_dialogue_training.pt",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        print(f"DIALOGUE_MODEL_EXPORTED={model_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
