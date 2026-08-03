# Human-evaluation log format

Store one row per pair and annotator in an anonymized CSV with columns:

```text
pair_id,annotator_id,q1,q2
```

After undoing randomized left/right display order, encode both answers as
`emf`, `flrs`, or `tie`. Q1 is overall fluency/coherence/naturalness; Q2 is
less undesirable phrase-level repetition. Do not include annotator names or
other identifying information.

Recompute vote totals, percentages, and Fleiss' kappa with:

```bash
python scripts/evaluate_human_preferences.py \
  --input human_eval/annotations.csv \
  --output human_eval/summary.json
```

The defaults validate the paper protocol of 200 pairs and six annotators.
The released template contains no fabricated or placeholder judgments.
