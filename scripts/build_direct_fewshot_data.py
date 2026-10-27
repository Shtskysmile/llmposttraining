#!/usr/bin/env python3
"""Build a direct-reasoning evaluation set using the tool baseline's examples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_prompts import TOOL_DEMONSTRATIONS


DIRECT_REASONING = {
    "From 1 through 8, choose 2 numbers, no two consecutive, with sum congruent to 1 modulo 3. How many choices?":
        ("7", "The valid nonconsecutive pairs whose sums are 1 modulo 3 are "
         "(1,3), (1,6), (2,5), (2,8), (3,7), (4,6), and (5,8). There are 7."),
    "A bag has 3 red and 2 blue balls. Two are drawn without replacement. Given at least one is red, what is the probability exactly one is red?":
        ("2/3", "There are choose(5,2) - choose(2,2) = 9 draws with at least one red. "
         "Exactly one red occurs in 3*2 = 6 draws, so the probability is 6/9 = 2/3."),
    "Find the least nonnegative x with x congruent to 2 modulo 3 and x congruent to 3 modulo 5.":
        ("8", "The value 8 is 2 modulo 3 and 3 modulo 5. Since the moduli are coprime, "
         "all solutions repeat every 15, so 8 is the least nonnegative solution."),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.input)
    output_path = Path(args.output)
    source_bytes = source_path.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()
            if line.strip()]
    examples = []
    for question, _tool_code, answer in TOOL_DEMONSTRATIONS:
        if question not in DIRECT_REASONING:
            raise SystemExit(f"no direct demonstration matches tool example: {question}")
        expected_answer, reasoning = DIRECT_REASONING[question]
        if answer != expected_answer:
            raise SystemExit(f"direct demonstration answer mismatch for: {question}")
        examples.append(
            f"Example problem: {question}\n"
            f"Direct reasoning: {reasoning}\n"
            f"Exact answer: <answer>{answer}</answer> \\boxed{{{answer}}}"
        )
    prefix = (
        "Solve the examples by direct mathematical reasoning without using tools. "
        "Then solve the final problem.\n\n"
        + "\n\n".join(examples)
        + "\n\nFinal problem:\n"
    )
    output_rows = [{**row, "prompt": prefix + row["prompt"]} for row in rows]
    output_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in output_rows).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_bytes)
    manifest = {
        "generator": str(Path(__file__).relative_to(Path.cwd())),
        "source": str(source_path),
        "source_sha256": sha256(source_bytes),
        "output_sha256": sha256(output_bytes),
        "n": len(output_rows),
        "prompt_protocol": "direct reasoning; same three example questions as tool demonstrations",
        "examples": [
            {"question": question, "answer": answer}
            for question, _code, answer in TOOL_DEMONSTRATIONS
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({"output": str(output_path), "manifest": str(manifest_path),
                      "n": len(output_rows), "sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
