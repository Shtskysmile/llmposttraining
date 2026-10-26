#!/usr/bin/env python3
"""Generate exact-answer tool-math problems with rule-family held-out splits."""
import argparse
import hashlib
import itertools
import json
import math
import random
from fractions import Fraction
from pathlib import Path


def _count_nonadjacent(n, k, modulus, residue):
    return sum(sum(xs) % modulus == residue for xs in itertools.combinations(range(1, n + 1), k)
               if all(b - a > 1 for a, b in zip(xs, xs[1:])))


def _crt(a, m, b, n):
    for x in range(math.lcm(m, n)):
        if x % m == a and x % n == b:
            return x
    raise ValueError("No solution")


def build(seed=5090, train_size=300, validation_size=60, test_size=90):
    rng = random.Random(seed)
    rows = {s: [] for s in ("train", "validation", "test")}
    seen_instances = set()
    seen_prompts = set()
    # Rule shapes are intentionally disjoint: test uses different bounds and
    # probability asks the complementary conditional event.
    for split, size in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        attempts = 0
        while len(rows[split]) < size:
            i = len(rows[split])
            attempts += 1
            if attempts > size * 100:
                raise RuntimeError(f"Could not generate enough unique {split} instances.")
            family = i % 3
            if family == 0:
                n, k, mod = ((rng.randint(10, 22), rng.randint(3, 6), rng.randint(3, 11)) if split != "test"
                             else (rng.randint(23, 30), rng.randint(4, 7), rng.randint(7, 13)))
                residue = rng.randrange(mod)
                instance_key = f"count:{n}:{k}:{mod}:{residue}"
                answer = _count_nonadjacent(n, k, mod, residue)
                prompt = rng.choice([
                    (f"From the integers 1 through {n}, choose {k} distinct numbers so that no two are "
                     f"consecutive and their sum is congruent to {residue} modulo {mod}. How many choices are there?"),
                    (f"How many {k}-element subsets of 1 through {n} contain no consecutive integers and have "
                     f"a sum congruent to {residue} modulo {mod}?"),
                    (f"Select {k} distinct integers from 1..{n}, with every two selected values separated by "
                     f"at least one integer. How many selections have total congruent to {residue} mod {mod}?"),
                    (f"Count the ways to pick {k} numbers between 1 and {n} inclusive, with no consecutive pair "
                     f"and with the sum leaving remainder {residue} on division by {mod}."),
                ])
                kind = "constrained_counting"
                solution = f"Enumerate combinations, keep tuples with every adjacent gap > 1 and sum modulo {mod} equal to {residue}; count them."
                tool_code = (f"choices = [xs for xs in combinations(range(1, {n + 1}), {k}) "
                             f"if all(y - x > 1 for x, y in zip(xs, xs[1:])) "
                             f"and sum(xs) % {mod} == {residue}]\nprint(len(choices))")
            elif family == 1:
                red, blue = rng.randint(3, 40), rng.randint(3, 40)
                if split == "test":
                    red, blue = rng.randint(41, 60), rng.randint(41, 60)
                # Training asks P(two red | at least one red); test asks
                # P(exactly one red | at least one red), a held-out target rule.
                den = math.comb(red + blue, 2) - math.comb(blue, 2)
                num = math.comb(red, 2) if split != "test" else red * blue
                answer = str(Fraction(num, den))
                event = "both balls are red" if split != "test" else "exactly one ball is red"
                instance_key = f"prob:{red}:{blue}:{event}"
                probability_templates = [
                    (f"A bag has {red} red and {blue} blue balls. Two are drawn uniformly without replacement. "
                     f"Given that at least one drawn ball is red, what is the probability that {event}? "
                     "Give an exact fraction."),
                    (f"An urn contains {red} red and {blue} blue balls. Two are sampled without replacement. "
                     f"If at least one sampled ball is red, what is the conditional probability that {event}? "
                     "Return the exact fraction."),
                    (f"Two balls are selected uniformly without replacement from {red} red and {blue} blue balls. "
                     f"Assuming the selection includes a red ball, calculate the chance that {event}."),
                ]
                if split == "test":
                    probability_templates = [
                        (f"An urn contains {red} red and {blue} blue balls. Two are sampled uniformly without "
                         f"replacement. Conditioned on at least one red being drawn, find the probability that {event}. "
                         "Give a reduced fraction."),
                        (f"There are {red} red balls and {blue} blue balls. Draw two at random without replacement. "
                         f"Knowing that at least one is red, what is the probability that {event}?"),
                    ]
                prompt = rng.choice(probability_templates)
                kind = "conditional_probability"
                solution = "Count unordered pairs in the conditioning event and divide favorable pairs by that count; reduce the fraction."
                tool_code = (f"favorable = {'comb(' + str(red) + ', 2)' if split != 'test' else str(red * blue)}\n"
                             f"conditioning = comb({red + blue}, 2) - comb({blue}, 2)\n"
                             "g = gcd(favorable, conditioning)\n"
                             "print(f'{favorable // g}/{conditioning // g}')")
            else:
                if split == "test":
                    # Hold out non-coprime moduli while keeping residues consistent.
                    m, n = rng.choice([(8, 12), (9, 15), (10, 14), (12, 18)])
                else:
                    while True:
                        m, n = rng.sample(range(3, 31), 2)
                        if math.gcd(m, n) == 1:
                            break
                hidden_solution = rng.randrange(math.lcm(m, n))
                a, b = hidden_solution % m, hidden_solution % n
                constraints = tuple(sorted(((m, a), (n, b))))
                instance_key = f"crt:{constraints}"
                answer = str(_crt(a, m, b, n))
                prompt = rng.choice([
                    f"Find the least nonnegative integer x satisfying x ≡ {a} (mod {m}) and x ≡ {b} (mod {n}).",
                    f"What is the smallest x ≥ 0 with remainder {a} when divided by {m} and remainder {b} when divided by {n}?",
                    f"Solve x = {a} modulo {m} and x = {b} modulo {n}; report the least nonnegative solution.",
                    f"Find the least residue x in [0, lcm({m}, {n})) that satisfies x mod {m} = {a} and x mod {n} = {b}.",
                ])
                kind = "modular_system"
                solution = "Search one complete least-common-multiple period or combine the congruences with the Chinese remainder theorem."
                tool_code = (f"solutions = [x for x in range(lcm({m}, {n})) "
                             f"if x % {m} == {a} and x % {n} == {b}]\nprint(min(solutions))")
            if instance_key in seen_instances or prompt in seen_prompts:
                continue
            seen_instances.add(instance_key)
            seen_prompts.add(prompt)
            rows[split].append({"id": f"{split}-{i:05d}", "prompt": prompt, "answer": answer,
                                "kind": kind, "instance_key": instance_key,
                                "solution_outline": solution, "tool_code": tool_code})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tool_math_scale_seed5090")
    p.add_argument("--seed", type=int, default=5090)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--validation-size", type=int, default=300)
    p.add_argument("--test-size", type=int, default=300)
    a = p.parse_args()
    if min(a.train_size, a.validation_size, a.test_size) < 1:
        p.error("split sizes must be positive")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in build(a.seed, a.train_size, a.validation_size, a.test_size).items():
        path = out / f"{name}.jsonl"
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        path.write_bytes(payload)
        print(f"{name}: {len(rows)} rows -> {path}")
    manifest = {
        "generator": "data/generate_tool_math.py",
        "seed": a.seed,
        "split_sizes": {"train": a.train_size, "validation": a.validation_size, "test": a.test_size},
        "families": ["constrained_counting", "conditional_probability", "modular_system"],
        "split_design": {
            "counting": "test bounds and parameter ranges are disjoint from train/validation",
            "conditional_probability": "test asks exactly-one-red; train/validation ask both-red; counts are disjoint",
            "modular_system": "train/validation use coprime moduli; test uses consistent non-coprime moduli",
            "duplicate_instances_rejected_across_splits": True,
        },
        "files": {name: hashlib.sha256((out / f"{name}.jsonl").read_bytes()).hexdigest()
                  for name in ("train", "validation", "test")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
