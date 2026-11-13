"""Strict answer extraction and deterministic *process* verification.

This module extends the final-answer verifier used by the TRL implementation
(`src/verifier.py`) with a proof-carrying process checker for integer
arithmetic.  The process checker does not execute model-generated Python.  It

1. parses the known gold expression into an AST and enumerates the canonical
   intermediate values (post-order evaluation),
2. extracts arithmetic equations written by the model (``a op b = c``),
3. accepts a step only if the arithmetic is exact *and* both operands are
   grounded in literals from the prompt or results of earlier accepted steps,
4. reports how much of the canonical derivation the model actually proved.

Because coverage is measured against *gold* intermediate values, spamming
trivial true equations (``1 + 1 = 2``) cannot earn process reward.  This makes
the dense signal resistant to the usual arithmetic reward hacks.

The final-answer extractor is behaviour-compatible with the TRL project so the
two implementations can be compared directly.
"""
from __future__ import annotations

import ast
import bisect
import re
from fractions import Fraction

# ---------------------------------------------------------------------------
# Final answer extraction (same rules as src/verifier.py)
# ---------------------------------------------------------------------------
ANSWER_RE = re.compile(r"<answer>\s*([^<>\n]+?)\s*</answer>", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?$")
DEC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
EQUATION_RE = re.compile(rf"^\s*({DEC})\s*([+*×x÷/\-])\s*({DEC})\s*=\s*({DEC})\s*$")


def _number(token: str):
    token = token.strip().replace(",", "")
    if not NUM_RE.fullmatch(token):
        return None
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text: str):
    """Return the strict final answer as a Fraction, or None."""
    matches = ANSWER_RE.findall(text)
    if matches:
        # A malformed or repeated answer tag is never rescued by nearby prose.
        return _number(matches[0]) if len(matches) == 1 else None

    boxed = BOXED_RE.findall(text)
    if boxed:
        return _number(boxed[0]) if len(boxed) == 1 else None

    # Also accept a bare numeric final line; this is an unambiguous answer, but
    # format_error() still reports the missing required tag for analysis.
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1].strip().strip("$`*_").strip()
    if not candidate:
        if len(lines) == 1:
            return None
        candidate = lines[-2].strip().strip("$`*_").strip()
    value = _number(candidate)
    if value is not None:
        return value

    # Accept a standalone, correct arithmetic equality as a verifiable final
    # answer. Do not scrape numbers from explanations or keyword phrases.
    eq = EQUATION_RE.fullmatch(candidate)
    if eq:
        lhs, op, rhs, answer = eq.groups()
        try:
            a, b, stated = Fraction(lhs), Fraction(rhs), Fraction(answer)
            calculated = {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "×": lambda: a * b,
                "x": lambda: a * b,
                "/": lambda: a / b,
                "÷": lambda: a / b,
            }[op]()
            return stated if calculated == stated else None
        except (ValueError, ZeroDivisionError):
            return None
    return None


def normalize_gold(gold: str):
    return _number(str(gold))


def answer_score(completion: str, gold: str) -> float:
    pred, target = extract_answer(completion), normalize_gold(gold)
    return float(pred is not None and target is not None and pred == target)


def score(completion: str, gold: str) -> float:  # backwards-compatible name
    return answer_score(completion, gold)


def format_error(completion: str) -> bool:
    matches = ANSWER_RE.findall(completion)
    return len(matches) != 1 or _number(matches[0]) is None


def has_answer_tag(completion: str) -> bool:
    """True iff exactly one well-formed ``<answer>...</answer>`` tag exists."""
    return not format_error(completion)


# ---------------------------------------------------------------------------
# Expression parsing for the two data templates
# ---------------------------------------------------------------------------
_SYMBOLIC_RE = re.compile(r"Compute exactly:\s*(.*?)\.\s*Return", re.IGNORECASE | re.DOTALL)
_WORD_RE = re.compile(
    r"starts with\s+(-?\d+)\s+units,\s*produces\s+(-?\d+)\s+units in each of\s+(-?\d+)\s+batches,"
    r"\s*then removes\s+(-?\d+)\s+defective units",
    re.IGNORECASE | re.DOTALL,
)


def expression_from_prompt(prompt: str) -> str | None:
    """Extract a canonical integer expression from a supported prompt template."""
    match = _SYMBOLIC_RE.search(prompt)
    if match:
        return match.group(1).strip().rstrip(".")

    match = _WORD_RE.search(prompt)
    if match:
        start, per_batch, batches, discarded = match.groups()
        # Multiplication is written first so the AST matches the story order.
        return f"{start} + {per_batch} * {batches} - {discarded}"
    return None


class _UnsupportedExpression(ValueError):
    pass


def _eval_node(node: ast.AST) -> tuple[int, list[int]]:
    """Evaluate an integer AST, returning (value, post-order internal values)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value, []
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value, internals = _eval_node(node.operand)
        return -value, internals
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left, left_internal = _eval_node(node.left)
        right, right_internal = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        else:
            value = left * right
        return value, left_internal + right_internal + [value]
    raise _UnsupportedExpression(
        f"Unsupported node in gold expression: {type(node).__name__}"
    )


def _literals(node: ast.AST) -> list[int]:
    values: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, int):
            values.append(int(child.value))
    return values


def canonical_derivation(expression: str) -> dict:
    """Return the gold value, prompt literals, and canonical intermediate values."""
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise _UnsupportedExpression(f"Invalid gold expression {expression!r}") from exc
    value, internal = _eval_node(tree)
    return {
        "expression": expression,
        "value": value,
        "literals": sorted(set(_literals(tree))),
        # The final value is the last internal value; keep only intermediates.
        "intermediate_values": internal[:-1],
        "all_internal_values": internal,
    }


# ---------------------------------------------------------------------------
# Process extraction
# ---------------------------------------------------------------------------
_NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
_OP = r"[+\-*×x·÷/]"
_PROCESS_EQ_RE = re.compile(
    rf"\(?\s*({_NUM})\s*\)?\s*({_OP})\s*\(?\s*({_NUM})\s*\)?\s*=\s*\(?\s*({_NUM})\s*\)?"
)
_OP_NORMALIZE = {"×": "*", "x": "*", "·": "*", "*": "*", "÷": "/", "/": "/", "+": "+", "-": "-"}


def _to_int(token: str) -> int:
    return int(token.replace(",", ""))


def _apply(op: str, a: int, b: int) -> int | None:
    op = _OP_NORMALIZE.get(op, op)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


_LATEX_REPLACEMENTS = (
    ("−", "-"),  # U+2212 unicode minus
    ("\\times", "*"),
    ("\\cdot", "*"),
    ("\\div", "/"),
    ("\\left", ""),
    ("\\right", ""),
    ("\\(", "("),
    ("\\)", ")"),
    ("\\[", "["),
    ("\\]", "]"),
    ("\\,", ""),
    ("\\;", ""),
    ("\\!", ""),
    ("{,}", ","),
    ("**", "*"),
)


def _normalize_latex_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalise LaTeX spellings while mapping every output char to its input.

    The offset map lets the token-level reward manager place credit on the
    original response tokens even though replacements change string length.
    """
    current = text
    offsets = list(range(len(text))) + [len(text)]
    for old, new in _LATEX_REPLACEMENTS:
        if old not in current:
            continue
        out_chars: list[str] = []
        out_offsets: list[int] = []
        i = 0
        while i < len(current):
            if current.startswith(old, i):
                for char in new:
                    out_chars.append(char)
                    out_offsets.append(offsets[i])
                i += len(old)
            else:
                out_chars.append(current[i])
                out_offsets.append(offsets[i])
                i += 1
        current = "".join(out_chars)
        offsets = out_offsets + [len(text)]
    return current, offsets


def _normalize_latex(text: str) -> str:
    """Normalise common LaTeX/Markdown spellings of arithmetic equations."""
    return _normalize_latex_with_offsets(text)[0]


_RUN_RE = re.compile(r"[0-9+\-*/(),.= ×·÷−]+")


def _eval_numeric(node: ast.AST):
    """Evaluate a strict integer/float arithmetic AST, returning int or None."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_numeric(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_numeric(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_numeric(node.left)
        right = _eval_numeric(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0 or left % right != 0:
            return None
        return left // right
    return None


_STEP_LABEL_RE = re.compile(r"^\s*\d+[.)]\s+")


def _strip_step_label(text: str) -> str:
    return _STEP_LABEL_RE.sub("", text.strip(), count=1)


def _normalize_arithmetic_text(text: str) -> str:
    text = text.replace("−", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 1,234 -> 1234
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", text)
    return text


def _parse_int_expression(text: str):
    text = _normalize_arithmetic_text(_strip_step_label(text))
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    return _eval_numeric(tree)


def _leaf_values(expression: str) -> list[int] | None:
    """Return the numeric operands of an expression, applying unary minus.

    ``-338 * 12`` must yield ``[-338, 12]``, not ``[338, 12]``; otherwise a
    negative intermediate value produced by the previous step is not grounded.
    """
    try:
        tree = ast.parse(
            _normalize_arithmetic_text(_strip_step_label(expression)), mode="eval"
        ).body
    except SyntaxError:
        return None

    def collect(node):
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return [value]
            if isinstance(value, float) and value.is_integer():
                return [int(value)]
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant):
                inner = collect(operand)
                return None if inner is None else [-v for v in inner]
            return collect(operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return collect(node.operand)
        if isinstance(node, ast.BinOp):
            left = collect(node.left)
            right = collect(node.right)
            if left is None or right is None:
                return None
            return left + right
        return None

    return collect(tree)


def extract_derivation_chains(text: str) -> list[dict]:
    """Extract arithmetic equality chains with original-text end offsets.

    A chain is a maximal run of arithmetic characters such as
    ``552 * 866 = 552 * (800+60+6) = ... = 478032``.  Every part is evaluated
    exactly; the chain is a valid derivation step if all parts are equal.  The
    leaves of the leftmost part are the step's operands and are what grounding
    checks.  This handles place-value decompositions and multi-equality lines
    that a flat ``a op b = c`` regex would reject.
    """
    normalized, offsets = _normalize_latex_with_offsets(text)
    # Every post-processing step below preserves the string length, so the
    # offset map computed above stays valid for token-level reward placement.
    normalized = re.sub(
        r"\\text\{[^{}]*\}", lambda m: " " * len(m.group(0)), normalized
    )
    normalized = normalized.replace("$$", "  ").replace("$", " ")
    normalized = re.sub(
        r"(?m)^\s*[-*]\s+", lambda m: " " * len(m.group(0)), normalized
    )  # markdown bullets

    chains: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for match in _RUN_RE.finditer(normalized):
        run = match.group(0).strip()
        if "=" not in run:
            continue
        parts = [part.strip() for part in run.split("=") if part.strip()]
        if len(parts) < 2:
            continue
        values = [_parse_int_expression(part) for part in parts]
        if any(value is None for value in values):
            continue
        leaves = _leaf_values(parts[0])
        if leaves is None:
            continue
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        original_end = offsets[match.end()] if match.end() < len(offsets) else len(text)
        chains.append(
            {
                "leaves": leaves,
                "values": values,
                "result": values[-1],
                "equal": len(set(values)) == 1,
                "char_end": original_end,
            }
        )
    return chains


def extract_equations(text: str) -> list[dict]:
    """Extract arithmetic equations in reading order, de-duplicated by content.

    ``char_end`` is an offset into the *original* completion so the reward
    manager can align verified steps with response tokens.
    """
    normalized, offsets = _normalize_latex_with_offsets(text)
    equations: list[dict] = []
    seen: set[tuple[int, str, int, int]] = set()
    for match in _PROCESS_EQ_RE.finditer(normalized):
        a, op, b, c = match.groups()
        a, b, c = _to_int(a), _to_int(b), _to_int(c)
        key = (a, _OP_NORMALIZE.get(op, op), b, c)
        if key in seen:
            continue
        seen.add(key)
        original_end = offsets[match.end()] if match.end() < len(offsets) else len(text)
        equations.append(
            {"a": a, "op": op, "b": b, "c": c, "char_end": original_end}
        )
    return equations


def verify_process(expression: str | None, completion: str, gold: str | None = None) -> dict:
    """Verify a model completion against the gold expression.

    Returns public metrics plus ``verified_char_ends`` (positions of accepted
    steps) for token-level reward placement.
    """
    metrics = {
        "process_coverage": 0.0,
        "process_valid_steps": 0,
        "process_invalid_steps": 0,
        "process_ungrounded_steps": 0,
        "process_total_steps": 0,
        "process_invalid_rate": 0.0,
        "process_gold_ok": 0.0,
        "verified_char_ends": [],
    }
    if not expression:
        return metrics

    try:
        gold_info = canonical_derivation(expression)
    except _UnsupportedExpression:
        return metrics

    if gold is not None:
        try:
            metrics["process_gold_ok"] = float(gold_info["value"] == int(str(gold).strip()))
        except (TypeError, ValueError):
            metrics["process_gold_ok"] = 0.0

    chains = extract_derivation_chains(completion)
    metrics["process_total_steps"] = len(chains)
    grounded = set(gold_info["literals"])
    valid_values: set[int] = set()
    verified_ends: list[int] = []

    invalid = ungrounded = valid = 0
    for chain in chains:
        arithmetic_ok = bool(chain["equal"])
        grounded_ok = bool(chain["leaves"]) and all(
            leaf in grounded for leaf in chain["leaves"]
        )
        if arithmetic_ok and grounded_ok:
            valid += 1
            grounded.add(chain["result"])
            valid_values.add(chain["result"])
            verified_ends.append(chain["char_end"])
        elif arithmetic_ok:
            ungrounded += 1
        else:
            invalid += 1

    intermediates = gold_info["intermediate_values"]
    if intermediates:
        covered = sum(1 for value in intermediates if value in valid_values)
        metrics["process_coverage"] = covered / len(intermediates)
    metrics["process_valid_steps"] = valid
    metrics["process_invalid_steps"] = invalid
    metrics["process_ungrounded_steps"] = ungrounded
    metrics["process_invalid_rate"] = invalid / max(1, len(chains))
    metrics["verified_char_ends"] = verified_ends
    return metrics


def loop_repetition_ratio(text: str, n: int = 40) -> float:
    """Fraction of the completion covered by its most repeated character n-gram.

    The base Qwen3-0.6B sometimes degenerates into digit loops such as
    ``982 × 451 = 441, 982 × 451 = 441, ...``.  A 40-character repeated chunk
    is a strong signal of degeneration; normal arithmetic traces contain no
    such repetition.  Returns 0 when the longest repeated chunk occurs fewer
    than three times.
    """
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) < n * 2:
        return 0.0
    counts: dict[str, int] = {}
    for start in range(0, len(compact) - n + 1):
        chunk = compact[start : start + n]
        counts[chunk] = counts.get(chunk, 0) + 1
    max_count = max(counts.values(), default=0)
    if max_count < 3:
        return 0.0
    return min(1.0, max_count * n / len(compact))


def process_reward(
    prompt: str | None,
    completion: str,
    gold: str,
    *,
    expression: str | None = None,
    final_weight: float = 1.0,
    process_weight: float = 0.5,
    format_weight: float = 0.1,
    invalid_penalty: float = 0.05,
    loop_penalty_weight: float = 0.0,
) -> dict:
    """Compose a dense reward: final answer + proven process + format.

    ``process_component`` is returned separately so a token-level reward
    manager can place it on the exact tokens where verified steps complete
    while keeping the total reward identical to the scalar computation.
    """
    final_correct = answer_score(completion, gold)
    format_ok = float(has_answer_tag(completion))
    if expression is None and prompt:
        expression = expression_from_prompt(prompt)
    proc = verify_process(expression, completion, gold)

    process_component = process_weight * proc["process_coverage"]
    loop_ratio = loop_repetition_ratio(completion)
    loop_penalty = loop_penalty_weight * loop_ratio
    final_component = (
        final_weight * final_correct
        + format_weight * format_ok
        - invalid_penalty * proc["process_invalid_rate"]
        - loop_penalty
    )
    score_value = final_component + process_component

    # Reward dicts are averaged into metrics by both verl and the demo harness,
    # so every value must be numeric.  Do not add the expression string here.
    result = {
        "score": score_value,
        "acc": final_correct,
        "correct": final_correct,
        "format_ok": format_ok,
        "process_component": process_component,
        "final_component": final_component,
        "loop_ratio": loop_ratio,
        "loop_penalty": loop_penalty,
    }
    result.update({k: v for k, v in proc.items() if k != "verified_char_ends"})
    result["verified_char_ends"] = proc["verified_char_ends"]
    return result


# ---------------------------------------------------------------------------
# Token alignment helper used by the token-level reward manager
# ---------------------------------------------------------------------------
def char_ends_to_token_indices(tokenizer, token_ids: list[int], char_ends: list[int]) -> list[int]:
    """Map character offsets in the decoded response to token indices.

    Qwen uses byte-level BPE; for the ASCII arithmetic text in this project,
    concatenating per-token decodes reproduces the full decode.  We fall back
    to the last token if alignment is ambiguous.
    """
    if not char_ends or not token_ids:
        return []
    pieces = []
    for token_id in token_ids:
        try:
            pieces.append(tokenizer.decode([int(token_id)]))
        except Exception:
            pieces.append("")
    cumulative = []
    total = 0
    for piece in pieces:
        total += len(piece)
        cumulative.append(total)
    indices = []
    for end in char_ends:
        index = bisect.bisect_left(cumulative, int(end))
        indices.append(min(index, len(cumulative) - 1))
    # Preserve order but drop duplicates caused by multi-token steps sharing an
    # approximate boundary.
    deduped = []
    for index in indices:
        if not deduped or deduped[-1] != index:
            deduped.append(index)
    return deduped
