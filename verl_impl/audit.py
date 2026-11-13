#!/usr/bin/env python3
"""Audit an evaluation dump and re-check the dumped metrics from scratch.

Every validation dump row contains the rendered prompt (``input``), the model
``output``, and the verifier metrics.  This script re-derives the gold
expression from the prompt, recomputes accuracy and process coverage, and
reports:

* any disagreement between dumped and recomputed metrics,
* full-proof / partial-proof / answer-only / wrong counts,
* sample completions for manual inspection.

Usage:
    python verl_impl/audit.py --dump-dir verl_impl/outputs/eval_process --samples 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verl_impl.verifier import (  # noqa: E402
    answer_score,
    canonical_derivation,
    expression_from_prompt,
    format_error,
    verify_process,
)


def read_dump(dump_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(dump_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    rows = read_dump(Path(args.dump_dir))
    if not rows:
        raise SystemExit("Empty dump.")

    categories = {"full_proof": [], "partial_proof": [], "answer_only": [], "wrong": []}
    mismatches = []
    for row in rows:
        prompt = row["input"]
        output = row["output"]
        expression = expression_from_prompt(prompt)
        if expression is None:
            mismatches.append({"reason": "no_expression", "input": prompt[:200]})
            continue
        gold_value = canonical_derivation(expression)["value"]
        gold = str(gold_value)
        acc = answer_score(output, gold)
        proc = verify_process(expression, output, gold)
        dumped_acc = float(row.get("acc", row.get("score", 0.0)))
        if abs(acc - dumped_acc) > 1e-6:
            mismatches.append(
                {
                    "reason": "acc_mismatch",
                    "dumped": dumped_acc,
                    "recomputed": acc,
                    "expression": expression,
                }
            )
        if abs(proc["process_coverage"] - float(row.get("process_coverage", 0.0))) > 1e-6:
            mismatches.append(
                {
                    "reason": "coverage_mismatch",
                    "dumped": row.get("process_coverage"),
                    "recomputed": proc["process_coverage"],
                    "expression": expression,
                }
            )

        if acc >= 0.5 and proc["process_coverage"] >= 1.0:
            categories["full_proof"].append(row)
        elif acc >= 0.5:
            categories["answer_only"].append(row)
        elif proc["process_coverage"] > 0.0:
            categories["partial_proof"].append(row)
        else:
            categories["wrong"].append(row)

    n = len(rows)
    print(f"\nAudited {n} completions from {args.dump_dir}")
    print(f"format_errors: {sum(format_error(row['output']) for row in rows)}/{n}")
    for name, items in categories.items():
        print(f"{name}: {len(items)} ({len(items) / n:.1%})")
    print(f"metric mismatches: {len(mismatches)}")
    for mismatch in mismatches[:5]:
        print("  ", mismatch)

    for name in ("partial_proof", "answer_only"):
        print(f"\n--- {name} samples ---")
        for row in categories[name][: args.samples]:
            print(f"[expr] {expression_from_prompt(row['input'])}")
            print(f"[acc={row.get('acc')} coverage={row.get('process_coverage')}]")
            print(row["output"][:800])
            print()


if __name__ == "__main__":
    main()
