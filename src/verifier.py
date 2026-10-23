"""Strict answer extraction and exact rational scoring for generated arithmetic."""
from fractions import Fraction
import re

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
    candidate = lines[-1].strip()
    # Permit common math/Markdown wrappers only when the remaining line itself
    # is numeric or a simple arithmetic equality.
    candidate = candidate.strip().strip("$`*_").strip()
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
            calculated = {"+": lambda: a + b, "-": lambda: a - b,
                          "*": lambda: a * b, "×": lambda: a * b,
                          "x": lambda: a * b, "/": lambda: a / b,
                          "÷": lambda: a / b}[op]()
            return stated if calculated == stated else None
        except (ValueError, ZeroDivisionError):
            return None
    return None


def normalize_gold(gold: str):
    return _number(str(gold))


def score(completion: str, gold: str) -> float:
    pred, target = extract_answer(completion), normalize_gold(gold)
    return float(pred is not None and target is not None and pred == target)


def format_error(completion: str) -> bool:
    matches = ANSWER_RE.findall(completion)
    return len(matches) != 1 or _number(matches[0]) is None
