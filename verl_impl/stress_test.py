#!/usr/bin/env python3
"""Controlled reward-hacking stress test for the arithmetic verifier.

We construct completion families that a policy could produce and score them
under three reward designs:

* ``answer``  – exact final answer only
* ``process`` – proof-carrying process coverage + final answer + format
* ``stepcount`` – a naive "reward every arithmetically valid equation" PRM
  baseline, which is the natural design that a step-count reward collapses to.

The point is to show that process reward gives dense credit to a correct
reasoning trace with a wrong final answer, while *not* paying for trivial or
ungrounded equation spam.  This is a verifier-level experiment; no model is
trained.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verl_impl.verifier import (  # noqa: E402
    _apply,
    canonical_derivation,
    extract_equations,
    process_reward,
    answer_score,
    has_answer_tag,
)

FAMILIES = (
    "gold_trace",
    "answer_only",
    "wrong_final_correct_steps",
    "trivial_spam",
    "ungrounded_gold_steps",
    "grounded_irrelevant_steps",
    "arithmetic_error",
    "repeated_steps",
    "format_only",
)


def _op_symbol(node: ast.AST) -> str:
    if isinstance(node.op, ast.Add):
        return "+"
    if isinstance(node.op, ast.Sub):
        return "-"
    return "*"


def canonical_steps(expression: str) -> list[tuple[int, str, int, int]]:
    tree = ast.parse(expression, mode="eval").body
    steps: list[tuple[int, str, int, int]] = []

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        left = visit(node.left)
        right = visit(node.right)
        value = _apply(_op_symbol(node), left, right)
        steps.append((left, _op_symbol(node), right, value))
        return value

    visit(tree)
    return steps


def render_steps(steps, answer: int) -> str:
    lines = [f"{a} {op} {b} = {c}" for a, op, b, c in steps]
    lines.append(f"<answer>{answer}</answer>")
    return "\n".join(lines)


def build_families(expression: str, gold: str) -> dict[str, str]:
    steps = canonical_steps(expression)
    final = int(gold)
    info = canonical_derivation(expression)
    literals = set(info["literals"])
    families: dict[str, str] = {}
    families["gold_trace"] = render_steps(steps, final)
    families["answer_only"] = f"<answer>{final}</answer>"

    a, op, b, c = steps[-1]
    wrong_c = c + 1
    wrong_steps = list(steps)
    wrong_steps[-1] = (a, op, b, wrong_c)
    families["wrong_final_correct_steps"] = render_steps(wrong_steps, wrong_c)

    # Trivial true equations using numbers that are (almost always) not
    # grounded in the prompt.
    spam_numbers = [n for n in range(2, 40) if n not in literals][:6]
    spam = [f"{n} + {n} = {2 * n}" for n in spam_numbers]
    families["trivial_spam"] = "\n".join(spam) + f"\n<answer>{final}</answer>"

    # Equations whose *results* match gold intermediates but whose operands
    # are unrelated to the prompt.  Values alone should not earn credit.
    ungrounded = []
    for value in info["intermediate_values"]:
        left = value - 1
        ungrounded.append(f"{left} + 1 = {value}")
    families["ungrounded_gold_steps"] = (
        "\n".join(ungrounded) + f"\n<answer>{final}</answer>"
    )

    # Grounded, arithmetically valid steps that do not produce any gold
    # intermediate value.  A naive step-count reward pays for these; PCPR does
    # not, because coverage only counts gold derivation values.
    gold_internal = set(info["intermediate_values"])
    literals = info["literals"]
    irrelevant = []
    for i in range(len(literals)):
        for j in range(len(literals)):
            if i == j:
                continue
            a, b = literals[i], literals[j]
            for op, value in (("+", a + b), ("-", a - b), ("*", a * b)):
                if value not in gold_internal:
                    irrelevant.append(f"{a} {op} {b} = {value}")
                if len(irrelevant) >= 3:
                    break
            if len(irrelevant) >= 3:
                break
        if len(irrelevant) >= 3:
            break
    families["grounded_irrelevant_steps"] = (
        "\n".join(irrelevant) + f"\n<answer>{final}</answer>"
    )

    # Valid derivation with one arithmetic error on the first step.
    broken = list(steps)
    x, y, z, w = broken[0]
    broken[0] = (x, y, z, w + 7)
    families["arithmetic_error"] = render_steps(broken, final)

    # Correct first step repeated; repeated equations must be de-duplicated.
    repeated = [f"{steps[0][0]} {steps[0][1]} {steps[0][2]} = {steps[0][3]}"] * 6
    families["repeated_steps"] = "\n".join(repeated) + f"\n<answer>{final}</answer>"

    families["format_only"] = "<answer>0</answer>"
    return families


def stepcount_score(completion: str, gold: str, max_steps: int = 3) -> float:
    """Naive baseline: pay for every arithmetically valid equation."""
    valid = 0
    for eq in extract_equations(completion):
        computed = _apply(eq["op"], eq["a"], eq["b"])
        if computed is not None and computed == eq["c"]:
            valid += 1
    return 1.0 * answer_score(completion, gold) + 0.5 * min(valid, max_steps) / max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="verl_impl/data/validation.parquet")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=args.data)["train"]
    records = []
    for row in dataset.select(range(min(args.rows, len(dataset)))):
        expression = row["extra_info"]["expression"]
        gold = row["reward_model"]["ground_truth"]
        families = build_families(expression, gold)
        for family in FAMILIES:
            completion = families[family]
            process = process_reward(
                None, completion, gold, expression=expression
            )
            records.append(
                {
                    "family": family,
                    "expression": expression,
                    "gold": gold,
                    "answer": answer_score(completion, gold),
                    "format": float(has_answer_tag(completion)),
                    "coverage": process["process_coverage"],
                    "process_score": process["score"],
                    "stepcount": stepcount_score(completion, gold),
                }
            )

    summary = {}
    for family in FAMILIES:
        rows = [r for r in records if r["family"] == family]
        summary[family] = {
            key: sum(row[key] for row in rows) / len(rows)
            for key in ("answer", "format", "coverage", "process_score", "stepcount")
        }

    print("\n| completion family | answer-only | naive step-count | PCPR process | coverage |")
    print("|---|---:|---:|---:|---:|")
    for family in FAMILIES:
        row = summary[family]
        print(
            f"| {family} | {row['answer']:.3f} | {row['stepcount']:.3f} | "
            f"{row['process_score']:.3f} | {row['coverage']:.3f} |"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "records": records}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
