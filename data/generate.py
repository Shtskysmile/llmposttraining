#!/usr/bin/env python3
"""Create deterministic multi-step arithmetic RLVR data with a template holdout."""
import argparse
import json
import random
from pathlib import Path


def build(seed: int = 2026):
    rng = random.Random(seed)
    train, val, test = [], [], []
    def expression_row():
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        c, d = rng.randint(12, 39), rng.randint(100, 9999)
        template = rng.randrange(4)
        if template == 0:
            expr, answer = f"({a} + {b}) * {c} - {d}", (a + b) * c - d
        elif template == 1:
            expr, answer = f"{a} * {b} + {c} - {d}", a * b + c - d
        elif template == 2:
            expr, answer = f"({a} - {b}) * {c} + {d}", (a - b) * c + d
        else:
            expr, answer = f"{a} * {b} - {c} + {d}", a * b - c + d
        return {"prompt": f"Compute exactly: {expr}. Return the final integer in <answer>...</answer>.",
                "answer": str(answer), "kind": "three_op_expression"}

    train = [expression_row() for _ in range(128)]
    val = [expression_row() for _ in range(32)]

    for i in range(48):
        start, per_batch = rng.randint(100, 999), rng.randint(100, 999)
        batches, discarded = rng.randint(12, 39), rng.randint(100, 9999)
        answer = start + per_batch * batches - discarded
        question = (f"A factory starts with {start} units, produces {per_batch} units in each of {batches} batches, "
                    f"then removes {discarded} defective units. How many remain? Return the final integer in <answer>...</answer>.")
        test.append({"prompt": question, "answer": str(answer), "kind": "heldout_word_problem"})
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(("train", "validation", "test"), build(args.seed)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} rows -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
