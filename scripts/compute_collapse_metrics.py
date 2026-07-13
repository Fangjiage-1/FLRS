"""Compute mode-collapse detection metrics for generated text samples."""
import argparse
import json
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    return text.strip().split()


def distinct_n(texts: List[str], n: int) -> float:
    """distinct-n: fraction of n-grams that are unique (higher = more diverse)."""
    all_ngrams = []
    for t in texts:
        tokens = tokenize(t)
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def max_repeat_ngram(text: str, n: int) -> int:
    """Maximum consecutive repeats of the same n-gram in a sample."""
    tokens = tokenize(text)
    if len(tokens) < 2 * n:
        return 0
    max_run = 0
    current_run = 1
    prev = None
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        if ng == prev:
            current_run += 1
        else:
            current_run = 1
        prev = ng
        max_run = max(max_run, current_run)
    return max_run


def avg_top_ngram_freq(texts: List[str], n: int, top_k: int = 5) -> float:
    """Average frequency of top-k n-grams across samples."""
    freqs = []
    for t in texts:
        tokens = tokenize(t)
        ngs = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngs:
            continue
        counter = Counter(ngs)
        top = counter.most_common(top_k)
        total_ngrams = max(len(ngs), 1)
        avg_top_freq = sum(c / total_ngrams for _, c in top) / top_k
        freqs.append(avg_top_freq)
    return sum(freqs) / max(len(freqs), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("generated", "")
            if text.strip():
                texts.append(text)

    texts = texts[:args.n_samples]  # sample first N for consistency
    label = f"[{args.label}] " if args.label else ""

    for n in [1, 2, 3, 4]:
        d = distinct_n(texts, n)
        print(f"{label}distinct-{n}: {d:.6f}")

    for n in [2, 3, 4, 5]:
        reps = [max_repeat_ngram(t, n) for t in texts]
        avg = sum(reps) / max(len(reps), 1)
        mx = max(reps)
        pct_repeat = sum(1 for r in reps if r >= 3) / max(len(reps), 1)
        print(f"{label}max-repeat-{n}gram: avg={avg:.1f}  max={mx}  pct>=3repeats={pct_repeat:.3f}")

    for n in [2, 3, 4]:
        tf = avg_top_ngram_freq(texts, n, top_k=3)
        print(f"{label}top3-{n}gram-avg-freq: {tf:.6f}")

    # Simple diversity: total unique words / total words
    all_tokens = [t for text in texts for t in tokenize(text)]
    unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)
    print(f"{label}unique-word-ratio: {unique_ratio:.6f}")


if __name__ == "__main__":
    main()
