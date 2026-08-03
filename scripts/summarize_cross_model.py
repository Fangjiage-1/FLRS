#!/usr/bin/env python
"""Summarize paired cross-model runs into per-method mean and population SD."""

import argparse
import json
import statistics
from pathlib import Path

from compute_repetition_metrics import repeated_span_coverage


def load_last_jsonl(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows[-1]


def method_from_name(name):
    if name.startswith("sde-"):
        return "Fixed Rollback"
    if name.startswith("flrs-"):
        return "FLRS-Linear"
    return None


def collect(root):
    records = []
    for metrics_path in sorted(root.glob("seed_*/*/metrics.jsonl")):
        run_dir = metrics_path.parent
        method = method_from_name(run_dir.name)
        if method is None:
            continue
        generated_paths = sorted(run_dir.glob("all_generated_*.jsonl"))
        if not generated_paths:
            raise FileNotFoundError(f"Missing generated JSONL beside {metrics_path}")
        texts = [
            row.get("generated", "")
            for row in (
                json.loads(line)
                for line in generated_paths[-1].read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        ]
        metrics = load_last_jsonl(metrics_path)
        records.append(
            {
                "seed": int(metrics_path.parents[1].name.removeprefix("seed_")),
                "method": method,
                "ppl": float(metrics["ppl"]),
                "entropy": float(metrics["mean_entropy"]),
                "rsc": repeated_span_coverage(texts),
            }
        )
    if not records:
        raise FileNotFoundError(f"No seed_*/<run>/metrics.jsonl files found under {root}")
    return records


def aggregate(records):
    result = {}
    for method in ("Fixed Rollback", "FLRS-Linear"):
        rows = [row for row in records if row["method"] == method]
        if not rows:
            continue
        result[method] = {"seeds": [row["seed"] for row in rows]}
        for key in ("ppl", "entropy", "rsc"):
            values = [row[key] for row in rows]
            result[method][key] = {
                "mean": statistics.fmean(values),
                "sd": statistics.pstdev(values),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Model output directory.")
    parser.add_argument("--model", required=True, help="Display label, e.g. ELF-B.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    records = collect(args.input)
    report = {"model": args.model, "runs": records, "summary": aggregate(records)}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
