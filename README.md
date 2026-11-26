# MCR — Mathematical-native Comprehension & Response

MCR is an original, from-scratch neural-symbolic research system for English conversation. It is deliberately **not** a pretrained language model, a next-token predictor, or a fine-tuned foundation model.

## Architecture contract

- All MCR weights are initialized randomly at the start of training.
- Text is represented as UTF-8 byte values. There is no BPE, WordPiece, SentencePiece, downloaded vocabulary, or pretrained embedding table.
- A non-autoregressive byte CNN + bidirectional GRU builds sentence vectors.
- A contrastive metric-learning objective aligns conversation contexts with their assistant replies.
- A three-way semantic-relation head learns entailment, neutral relation, and contradiction from sentence pairs.
- Chat replies are selected from a paired semantic memory by nearest-neighbour retrieval. There is no decoder and no autoregressive text generation.
- The default network has approximately 15.1 million trainable parameters.
- Both T4 GPUs train the same model with PyTorch DistributedDataParallel and NCCL gradient all-reduce. Each GPU has its own model replica; VRAM is not pooled.

This is an intentionally different architecture from a conventional generative LLM. UTF-8 bytes are only the numeric input representation; they are not a downloaded token vocabulary or a pretrained language model.

## What this version can and cannot do

The system learns a semantic metric, sentence-pair relation labels, and response selection over its training corpus. It can retrieve conversational replies when a new conversation context resembles training examples. It can expose its learned entailment/neutral/contradiction judgment through `/judge`.

It does **not** yet produce arbitrary novel sentences, perform unrestricted multi-step reasoning, or possess a complete grounded world model. A response retrieved from the corpus can be irrelevant even when its similarity score is high. Check held-out retrieval/NLI scores and manually inspect the response bank before treating it as useful. This is a real trainable mathematical architecture, not a claim of perfect or human-level understanding.

## Data sources

See [DATASETS.md](DATASETS.md). The pipeline uses conversation data and labeled sentence-pair relations only. It downloads no model weights, tokenizer files, pretrained embeddings, or generation checkpoints.

## Kaggle T4 ×2 training

1. Create a Kaggle Notebook.
2. Select **GPU T4 ×2** in Notebook Settings and enable Internet.
3. Copy the entire contents of [`KAGGLE_ONE_CELL.py`](KAGGLE_ONE_CELL.py) into one code cell and run it.
4. The cell clones the `math-native-v1` branch, checks both GPUs, prepares the datasets, launches two synchronized training workers, evaluates held-out data, and exports the trained artifacts.
5. Use Kaggle **Save Version** to preserve the outputs.

Default workload:

- Up to 120,000 conversation context/reply pairs.
- Up to 200,000 entailment/neutral/contradiction examples.
- 2 epochs, 32 examples per GPU, 384-byte sequence cap.
- Mixed FP16 compute, AdamW, cosine schedule, gradient clipping, and resumable checkpoints.
- Checkpoint: `/kaggle/working/MCR_OUTPUT/training_state.pt`.
- Final model: `/kaggle/working/MCR_OUTPUT/mcr_model.pt`.
- Retrieval memory: `/kaggle/working/MCR_OUTPUT/response_index.npy` and `response_bank.jsonl`.
- Export: `/kaggle/working/MCR_MATH_NATIVE.zip`.

The dataset caps and number of epochs can be changed in the Kaggle cell. A successful run is not the same thing as good comprehension; the report contains held-out metrics.

## Chat after training

```python
!python /kaggle/working/MCR/chat.py --model-dir /kaggle/working/MCR_OUTPUT --debug
```

Commands: `/reset`, `/exit`, `/matches`, and `/judge PREMISE || HYPOTHESIS`.

## Run the offline tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The offline tests do not download data or require CUDA. Training requires Kaggle's preinstalled PyTorch CUDA build plus the Hugging Face `datasets` reader.
