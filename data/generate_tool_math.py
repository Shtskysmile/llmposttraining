#!/usr/bin/env python3
"""Generate exact-answer tool-math problems with rule-family held-out splits."""
import argparse
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
    # Rule shapes are intentionally disjoint: test uses different bounds and
    # probability asks the complementary conditional event.
    for split, size in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        for i in range(size):
            family = i % 3
            if family == 0:
                n, k, mod = ((rng.randint(10, 18), rng.randint(3, 5), rng.randint(3, 7)) if split != "test"
                             else (rng.randint(20, 28), rng.randint(4, 6), rng.randint(5, 9)))
                residue = rng.randrange(mod)
                answer = _count_nonadjacent(n, k, mod, residue)
                prompt = (f"From the integers 1 through {n}, choose {k} distinct numbers so that no two are "
                          f"consecutive and their sum is congruent to {residue} modulo {mod}. How many choices are there?")
                kind = "constrained_counting"
                solution = f"Enumerate combinations, keep tuples with every adjacent gap > 1 and sum modulo {mod} equal to {residue}; count them."
                tool_code = (f"choices = [xs for xs in combinations(range(1, {n + 1}), {k}) "
                             f"if all(y - x > 1 for x, y in zip(xs, xs[1:])) "
                             f"and sum(xs) % {mod} == {residue}]\nprint(len(choices))")
            elif family == 1:
                red, blue = rng.randint(3, 9), rng.randint(3, 10)
                if split == "test":
                    red, blue = rng.randint(7, 15), rng.randint(6, 16)
                # Training asks P(two red | at least one red); test asks
                # P(exactly one red | at least one red), a held-out target rule.
                den = math.comb(red + blue, 2) - math.comb(blue, 2)
                num = math.comb(red, 2) if split != "test" else red * blue
                answer = str(Fraction(num, den))
                event = "both balls are red" if split != "test" else "exactly one ball is red"
                prompt = (f"A bag has {red} red and {blue} blue balls. Two are drawn uniformly without replacement. "
                          f"Given that at least one drawn ball is red, what is the probability that {event}? "
                          "Give an exact fraction.")
                kind = "conditional_probability"
                solution = "Count unordered pairs in the conditioning event and divide favorable pairs by that count; reduce the fraction."
                tool_code = (f"favorable = {'comb(' + str(red) + ', 2)' if split != 'test' else str(red * blue)}\n"
                             f"conditioning = comb({red + blue}, 2) - comb({blue}, 2)\n"
                             "g = gcd(favorable, conditioning)\n"
                             "print(f'{favorable // g}/{conditioning // g}')")
            else:
                if split == "test":
                    m, n = rng.choice([(7, 11), (8, 13), (9, 14), (11, 15)])
                else:
                    m, n = rng.choice([(3, 5), (4, 7), (5, 8), (6, 11)])
                a, b = rng.randrange(m), rng.randrange(n)
                answer = str(_crt(a, m, b, n))
                prompt = (f"Find the least nonnegative integer x satisfying x ≡ {a} (mod {m}) and "
                          f"x ≡ {b} (mod {n}).")
                kind = "modular_system"
                solution = "Search one complete least-common-multiple period or combine the congruences with the Chinese remainder theorem."
                tool_code = (f"solutions = [x for x in range(lcm({m}, {n})) "
                             f"if x % {m} == {a} and x % {n} == {b}]\nprint(min(solutions))")
            rows[split].append({"id": f"{split}-{i:05d}", "prompt": prompt, "answer": answer,
                                "kind": kind, "solution_outline": solution, "tool_code": tool_code})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tool_math")
    p.add_argument("--seed", type=int, default=5090)
    p.add_argument("--train-size", type=int, default=300)
    p.add_argument("--validation-size", type=int, default=60)
    p.add_argument("--test-size", type=int, default=90)
    a = p.parse_args()
    if min(a.train_size, a.validation_size, a.test_size) < 1:
        p.error("split sizes must be positive")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in build(a.seed, a.train_size, a.validation_size, a.test_size).items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
