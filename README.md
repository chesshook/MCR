# MCR — Conversational Model, Kaggle T4 ×2

MCR is a reproducible conversational-model fine-tuning project. It adapts an open 3B instruction model using a multi-source conversation mixture and synchronized dual-GPU QLoRA.

> **Architecture disclosure:** MCR v2 uses Hugging Face's open SmolLM3-3B pretrained instruction model as its foundation. The base model supplies broad language and world-knowledge capability; this repository trains an MCR-specific LoRA adapter. It does not pretrain a 3B foundation model from random weights, and it does not use Qwen. Training a comparable foundation model from scratch is not realistic in one Kaggle T4×2 session.

## Pipeline

- NF4 4-bit QLoRA, FP16 compute, rank-32 LoRA adapters.
- Two torchrun workers, one process per T4, using PyTorch DDP + NCCL all-reduce to synchronize gradients after each accumulation window. VRAM is not pooled.
- English training mixture: Smol-SmolTalk (50k cap), UltraChat 200k SFT (25k cap), and OpenAssistant OASST1 human conversation paths (10k cap).
- Native SmolLM3 chat template with strict assistant-only labels. Dataset preparation refuses to proceed if valid assistant-span masks are unavailable.
- 2,048-token training context, 2% held-out validation, resumable checkpoints every 500 optimizer steps, evaluation every 1,000 steps.
- One Kaggle cell clones this repo, prepares the data, runs both GPUs, saves the adapter, and records manual-review chat examples.

## Kaggle start

1. Create a Kaggle Notebook.
2. Set **Settings → Accelerator → GPU T4 ×2**.
3. Turn **Internet On**.
4. Paste the complete KAGGLE_ONE_CELL.py file into one notebook cell and run it.

The cell downloads this repository and its pinned dependencies. The training launcher writes:
- /kaggle/working/MCR_OUTPUT — adapter, tokenizer, checkpoints and report
- /kaggle/working/MCR_TOKENIZED_DATA — tokenized conversation mixture and manifest
- /kaggle/working/MCR_ADAPTER.zip — final adapter export without intermediate checkpoints

After completion, use Kaggle **Save Version** to preserve the outputs. To chat in the same notebook after training:

```python
!python /kaggle/working/MCR/chat.py --adapter-dir /kaggle/working/MCR_OUTPUT
```

## Important limitations

This is supervised fine-tuning of an existing instruction model, not from-scratch pretraining. It can adapt the model's conversational behaviour, but does not guarantee perfect understanding, factual correctness, or human-level reasoning. Broad language capability primarily comes from the pretrained foundation. Held-out loss and the generated smoke-test answers must be inspected; a successful training run alone is not proof of intelligence.

See DATASETS.md for data sources and license notes.
