# MCR data and model provenance

MCR builds an English conversational mixture from these sources. Source licenses are retained in the runtime manifest. Check each upstream dataset card before redistributing a trained derivative, especially because datasets can be updated independently of this code.

| Source | Use | Upstream license |
|---|---|---|
| [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Short, varied instruction and dialogue samples | Apache-2.0 on dataset card; verify upstream component terms |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (train_sft) | Multi-turn instruction conversations | MIT |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English parent-linked human conversation paths | Apache-2.0 |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Pretrained instruction foundation model | Apache-2.0 |

The code filters OASST1 to English, non-deleted, non-synthetic messages in ready-for-export trees, then follows each leaf to its parent chain. It does not import non-English OASST data in this first recipe.

Default sampling caps are 50,000 Smol-SmolTalk examples, 25,000 UltraChat examples and 10,000 OASST1 paths. The data is shuffled with fixed seeds, de-duplicated by normalized message content and split into 98% train / 2% validation. The caps are configurable in launch_kaggle.py.

## Important provenance distinction

SmolLM3-3B is already pretrained and instruction-tuned. Its public model card describes a 3B decoder-only model, 11.2T pretraining tokens and subsequent instruction/alignment stages. MCR does not recreate that enormous pretraining run. It fine-tunes an adapter on the listed conversational mixture. The base model's own license and attribution continue to apply.

The MCR repository contains code and provenance metadata only. It does not commit downloaded datasets, base-model weights, Kaggle caches or trained model weights.
