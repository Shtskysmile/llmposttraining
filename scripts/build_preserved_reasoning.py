#!/usr/bin/env python3
"""Build a base-correct common SFT subset with preserved reasoning targets."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import score


ANSWER_OPEN = re.compile(r"<answer\b[^>]*>", re.IGNORECASE)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def normalized_target(completion, answer):
    match = ANSWER_OPEN.search(completion)
    reasoning = completion[:match.start()] if match else completion
    lines = reasoning.rstrip().splitlines()
    if lines and score(lines[-1], answer):
        lines.pop()
    reasoning = "\n".join(lines).rstrip()
    if reasoning:
        reasoning += "\n"
    return reasoning + f"<answer>{answer}</answer>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="original prompt/gold JSONL")
    parser.add_argument("--teacher-run", required=True,
                        help="single-sample src/evaluate.py JSONL from the base model")
    parser.add_argument("--common-output", required=True,
                        help="same teacher-correct prompts for final and trace SFT")
    parser.add_argument("--preserved-output", required=True,
                        help="same prompts with verified base reasoning targets")
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    teacher_path = Path(args.teacher_run)
    teacher_config_path = teacher_path.with_suffix(".run_config.json")
    if not teacher_config_path.is_file():
        raise SystemExit(f"teacher run config is missing: {teacher_config_path}")
    teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
    source = read_jsonl(source_path)
    teacher = read_jsonl(teacher_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if len(source) != len(teacher):
        raise SystemExit(f"row count mismatch: {len(source)} source, {len(teacher)} teacher")
    if teacher_config.get("n") != len(source) or teacher_config.get("data_sha256") != source_sha256:
        raise SystemExit("teacher run config does not match the source dataset hash and row count")
    if teacher_config.get("samples_per_question") != 1:
        raise SystemExit("teacher run must contain exactly one completion per source question")

    common_rows, preserved_rows = [], []
    wrong = malformed = format_errors = 0
    for index, (src, generated) in enumerate(zip(source, teacher)):
        prompt, answer = src["prompt"], str(src["answer"])
        if generated.get("prompt") != prompt or str(generated.get("gold")) != answer:
            raise SystemExit(f"teacher output is not aligned with source row {index}")
        completions = generated.get("completions", [])
        if len(completions) != 1:
            raise SystemExit(f"teacher output must contain exactly one completion at row {index}")
        completion = completions[0]
        if not score(completion, answer):
            wrong += 1
            continue
        row_id = src.get("id", f"train-{index:04d}")
        common = {"id": row_id, "prompt": prompt, "answer": answer}
        common_rows.append(common)
        preserved_rows.append({
            **common,
            "target": normalized_target(completion, answer),
            "source_index": index,
        })
        format_errors += int(not bool(re.search(
            r"<answer>\s*[^<>\n]+\s*</answer>", completion, re.IGNORECASE)))

    if not common_rows:
        raise SystemExit("no base-correct rows remain for the controlled SFT comparison")
    for output_path, rows in ((args.common_output, common_rows),
                              (args.preserved_output, preserved_rows)):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8")

    audit = {
        "source": str(source_path),
        "source_sha256": source_sha256,
        "teacher_run": str(teacher_path),
        "teacher_run_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "teacher_run_config": teacher_config,
        "source_rows": len(source),
        "teacher_correct_rows_retained": len(common_rows),
        "teacher_incorrect_or_unparseable_rows_dropped": wrong,
        "teacher_format_error_rows_retained": format_errors,
        "common_output": args.common_output,
        "common_output_sha256": hashlib.sha256(Path(args.common_output).read_bytes()).hexdigest(),
        "preserved_output": args.preserved_output,
        "preserved_output_sha256": hashlib.sha256(Path(args.preserved_output).read_bytes()).hexdigest(),
        "selection_rule": "retain only source-aligned base completions that pass the exact verifier",
        "target_rule": "preserve reasoning before the first answer tag and append one normalized gold answer tag",
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
