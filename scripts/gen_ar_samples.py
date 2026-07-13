#!/usr/bin/env python
"""Generate unconditional text samples from autoregressive (AR) models for
baseline comparison with ELF.

Output format: JSONL with {"id": int, "generated": str} — identical to
what test_generation_uncond writes, so eval_ppl.py can consume it directly.

Example:
    python scripts/gen_ar_samples.py \
        --model gpt2 --num_samples 1000 --max_length 1024 \
        --output outputs/ar_baselines/gpt2_small.jsonl

    # Then evaluate:
    python scripts/eval_ppl.py --input outputs/ar_baselines/gpt2_small.jsonl
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Generate unconditional AR baseline samples")
    p.add_argument("--model", type=str, required=True,
                   help="HF model name (e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl)")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (default 1.0 for standard)")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus sampling top-p (default 1.0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_length", type=int, default=0,
                   help="Minimum length in tokens (pad if shorter)")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory to cache downloaded model weights")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_dir = args.cache_dir or None
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cache_dir=cache_dir,
    ).to(device).eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_texts = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    total_generated = 0

    pbar = tqdm(total=args.num_samples, desc=f"Generating ({args.model}, T={args.temperature})")
    for _ in range(num_batches):
        if total_generated >= args.num_samples:
            break
        current_bs = min(args.batch_size, args.num_samples - total_generated)

        input_ids = torch.full(
            (current_bs, 1), tokenizer.bos_token_id,
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_length,
                min_new_tokens=args.min_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i in range(outputs.shape[0]):
            if total_generated >= args.num_samples:
                break
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            all_texts.append(text)
            total_generated += 1
            pbar.update(1)

    pbar.close()

    with open(args.output, "w", encoding="utf-8") as f:
        for i, text in enumerate(all_texts):
            f.write(json.dumps({"id": i, "generated": text}, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_texts)} samples to {args.output}")


if __name__ == "__main__":
    main()
