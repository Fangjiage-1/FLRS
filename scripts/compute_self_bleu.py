"""Compute self-BLEU for a set of generated text samples.

Self-BLEU measures how similar generated samples are to each other.
High self-BLEU -> texts are repetitive -> mode collapse.
Low self-BLEU -> texts are diverse -> healthy generation.
"""
import argparse
import json
import math
import random
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization (no external deps)."""
    return text.strip().split()


def ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu(hypothesis: List[str], references: List[List[str]], max_n: int = 4):
    """Compute BLEU score for a single hypothesis against multiple references."""
    hyp_ngram_counts = {n: Counter(ngrams(hypothesis, n)) for n in range(1, max_n + 1)}
    ref_ngram_max = {n: Counter() for n in range(1, max_n + 1)}

    for ref in references:
        for n in range(1, max_n + 1):
            ref_ngram_max[n] |= Counter(ngrams(ref, n))

    # Clipped precision for each n
    precisions = {}
    for n in range(1, max_n + 1):
        hyp_counts = hyp_ngram_counts[n]
        ref_max = ref_ngram_max[n]
        clipped = sum(min(hyp_counts[ng], ref_max[ng]) for ng in hyp_counts)
        total = max(sum(hyp_counts.values()), 1)
        precisions[n] = clipped / total if total > 0 else 0.0

    # If any precision is zero, BLEU goes to around zero
    if any(p == 0.0 for p in precisions.values()):
        return 0.0

    # Geometric mean of n-gram precisions
    log_bleu = sum(math.log(p) for p in precisions.values()) / max_n

    # Brevity penalty
    hyp_len = len(hypothesis)
    ref_len = min(abs(len(ref) - hyp_len) for ref in references) + hyp_len  # closest ref len proxy
    # Simplified: use average ref length
    ref_lens = [len(ref) for ref in references]
    avg_ref_len = sum(ref_lens) / max(len(ref_lens), 1)
    bp = min(1.0, math.exp(1.0 - avg_ref_len / max(hyp_len, 1)))

    return bp * math.exp(log_bleu)


def compute_self_bleu(texts: List[str], n_samples: int = 500, max_n: int = 4) -> float:
    """Compute self-BLEU: average BLEU where each text is hypothesis, others are references.

    To keep computation manageable, we sample n_texts and compare each against
    a random subset of k other texts as references.
    """
    n = min(len(texts), n_samples)
    sampled = random.sample(texts, n)
    tokenized = [tokenize(t) for t in sampled]

    # For large n, don't do O(n^2). Sample pairs.
    n_pairs = min(n, 500)
    indices = list(range(n))
    random.shuffle(indices)
    hypothesis_indices = indices[:n_pairs]

    scores = []
    for i in hypothesis_indices:
        hyp = tokenized[i]
        # Use 20 random references
        ref_indices = [j for j in range(n) if j != i]
        if len(ref_indices) > 20:
            ref_indices = random.sample(ref_indices, 20)
        refs = [tokenized[j] for j in ref_indices]
        if len(hyp) < 4:
            continue
        bleu = sentence_bleu(hyp, refs, max_n=max_n)
        scores.append(bleu)

    return sum(scores) / max(len(scores), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to all_generated_*.jsonl")
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("generated", "")
            if text.strip():
                texts.append(text)

    print(f"Loaded {len(texts)} non-empty texts")

    seed = 42
    random.seed(seed)

    for max_n in [1, 2, 3, 4]:
        sb = compute_self_bleu(texts, n_samples=args.n_samples, max_n=max_n)
        print(f"Self-BLEU-{max_n}: {sb:.6f}")


if __name__ == "__main__":
    main()
