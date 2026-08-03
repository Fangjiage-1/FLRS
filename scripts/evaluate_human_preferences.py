#!/usr/bin/env python
"""Aggregate pairwise human judgments and compute Fleiss' kappa.

Input CSV columns:
    pair_id, annotator_id, q1, q2

The q1/q2 values must be model identities after undoing randomized display
order: ``emf``, ``flrs``, or ``tie``.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ("emf", "flrs", "tie")
REQUIRED_COLUMNS = ("pair_id", "annotator_id", "q1", "q2")


def fleiss_kappa(item_counts):
    """Compute Fleiss' kappa from equal-size item-by-category counts."""
    if not item_counts:
        raise ValueError("No items were supplied.")
    ratings_per_item = sum(item_counts[0])
    if ratings_per_item < 2:
        raise ValueError("Fleiss' kappa requires at least two ratings per item.")
    if any(sum(row) != ratings_per_item for row in item_counts):
        raise ValueError("Every pair must have the same number of ratings.")

    n_items = len(item_counts)
    observed = sum(
        (sum(value * value for value in row) - ratings_per_item)
        / (ratings_per_item * (ratings_per_item - 1))
        for row in item_counts
    ) / n_items
    category_totals = [
        sum(row[column] for row in item_counts)
        for column in range(len(LABELS))
    ]
    proportions = [
        total / (n_items * ratings_per_item)
        for total in category_totals
    ]
    expected = sum(value * value for value in proportions)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def read_annotations(path, expected_pairs, expected_annotators):
    grouped = {"q1": defaultdict(Counter), "q2": defaultdict(Counter)}
    seen = set()
    annotators = set()

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            pair_id = row["pair_id"].strip()
            annotator_id = row["annotator_id"].strip()
            if not pair_id or not annotator_id:
                raise ValueError(f"Line {line_number}: pair_id and annotator_id are required.")
            key = (pair_id, annotator_id)
            if key in seen:
                raise ValueError(f"Line {line_number}: duplicate rating for {key}.")
            seen.add(key)
            annotators.add(annotator_id)

            for question in ("q1", "q2"):
                label = row[question].strip().lower()
                if label not in LABELS:
                    raise ValueError(
                        f"Line {line_number}: {question} must be one of {LABELS}, got {label!r}."
                    )
                grouped[question][pair_id][label] += 1

    pair_ids = sorted(grouped["q1"])
    if expected_pairs is not None and len(pair_ids) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs, found {len(pair_ids)}.")
    if expected_annotators is not None and len(annotators) != expected_annotators:
        raise ValueError(
            f"Expected {expected_annotators} annotators, found {len(annotators)}."
        )
    for question in ("q1", "q2"):
        if set(grouped[question]) != set(pair_ids):
            raise ValueError(f"{question} does not contain the same pair IDs.")
        for pair_id in pair_ids:
            ratings = sum(grouped[question][pair_id].values())
            if expected_annotators is not None and ratings != expected_annotators:
                raise ValueError(
                    f"{question}, pair {pair_id}: expected {expected_annotators} ratings, "
                    f"found {ratings}."
                )
    return grouped, pair_ids, annotators


def summarize(grouped, pair_ids):
    report = {}
    for question in ("q1", "q2"):
        matrix = [
            [grouped[question][pair_id][label] for label in LABELS]
            for pair_id in pair_ids
        ]
        counts = {
            label: sum(row[index] for row in matrix)
            for index, label in enumerate(LABELS)
        }
        total = sum(counts.values())
        report[question] = {
            "counts": counts,
            "percentages": {
                label: round(100.0 * count / total, 1)
                for label, count in counts.items()
            },
            "fleiss_kappa": round(fleiss_kappa(matrix), 3),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Anonymized item-level annotation CSV.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--expected-pairs", type=int, default=200)
    parser.add_argument("--expected-annotators", type=int, default=6)
    args = parser.parse_args()

    grouped, pair_ids, annotators = read_annotations(
        args.input, args.expected_pairs, args.expected_annotators
    )
    report = {
        "n_pairs": len(pair_ids),
        "n_annotators": len(annotators),
        "n_judgments": len(pair_ids) * len(annotators),
        **summarize(grouped, pair_ids),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
