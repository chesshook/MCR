# MCR training data and provenance

MCR downloads datasets at run time. It does **not** download or use pretrained model weights, language tokenizers, or precomputed model embeddings.

| Dataset | Training role | License / use note |
|---|---|---|
| [DailyDialog](https://huggingface.co/datasets/daily_dialog) | Everyday multi-turn dialogue | Commonly distributed under CC BY-NC-SA 4.0; verify the current dataset card and source terms. |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Longer multi-turn assistant conversations | Dataset card lists MIT; verify component/source terms. |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English human conversation paths | Apache-2.0; verify the current dataset card and per-message provenance. |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | Entailment, neutral, contradiction sentence pairs | CC BY-SA 4.0; retain attribution. |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | Additional sentence-pair relation supervision | CC BY-SA 4.0; retain attribution. |

The loader skips sources that fail to load and records retained counts. Training stops if fewer than 1,000 usable conversation pairs or sentence-relation pairs are available. The runtime `data_manifest.json` records the sources and sample counts actually used. Review upstream terms before redistributing a trained model, especially the non-commercial conditions on DailyDialog.

## Labels

SNLI and MultiNLI labels are interpreted as: `0 = entailment`, `1 = neutral`, `2 = contradiction`. Invalid/unlabeled rows are excluded.

## Data split

Data is deduplicated for conversations, deterministically shuffled, and split into training/validation subsets. Validation examples are not used for gradient updates. The limits configured in the Kaggle launcher are maximums, not guaranteed retained totals.
