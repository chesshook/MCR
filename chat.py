"""Interactive MCR chat using the saved adapter and the same open base model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
SYSTEM_PROMPT = (
    "/no_think\nYou are MCR, a clear and thoughtful conversational assistant. "
    "Understand the full context before answering. Explain ideas in plain language. "
    "Be accurate, acknowledge uncertainty, and never invent facts or pretend to have performed actions. "
    "Use earlier conversation context when it is relevant."
)


def load_chat_model(adapter_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3B model's 4-bit chat runtime.")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def answer(model, tokenizer, history, user_text: str, max_new_tokens: int = 384) -> str:
    # Keep a bounded token window while retaining complete user/assistant pairs.
    context = list(history)
    while True:
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + context
            + [{"role": "user", "content": user_text}]
        )
        probe = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if probe.shape[-1] <= 4096 or len(context) < 2:
            break
        context = context[2:]

    inputs = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.65,
            top_p=0.90,
            repetition_penalty=1.06,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="/kaggle/working/MCR_OUTPUT")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    model, tokenizer = load_chat_model(args.adapter_dir)
    if args.smoke_test:
        prompts = [
            "Hi! Let's have a normal conversation. What would you like to talk about?",
            "The river bank was muddy while the bank manager counted the deposits. Explain the two meanings of bank in this sentence.",
            "Can you teach me how memory works? Explain it simply.",
            "I told Maya that she should bring her book. Who should bring the book?",
            "I did not say that the answer was impossible. What exactly did I deny?",
        ]
        results = []
        for prompt in prompts:
            print("\nUSER:", prompt, flush=True)
            response = answer(model, tokenizer, [], prompt, args.max_new_tokens)
            print("MCR:", response, flush=True)
            results.append({"prompt": prompt, "response": response})
        output_file = Path(args.adapter_dir) / "mcr_smoke_test.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Smoke-test responses saved to: {output_file}", flush=True)
        return

    history = []
    print("MCR chat ready. Commands: /exit, /reset", flush=True)
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit"}:
            break
        if prompt.lower() == "/reset":
            history = []
            print("Conversation cleared.")
            continue
        response = answer(model, tokenizer, history, prompt, args.max_new_tokens)
        print("\nMCR>", response, flush=True)
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])
        history = history[-20:]


if __name__ == "__main__":
    main()
