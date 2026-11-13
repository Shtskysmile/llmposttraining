"""Unit tests for the proof-carrying process verifier."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl_impl.verifier import (  # noqa: E402
    answer_score,
    canonical_derivation,
    expression_from_prompt,
    extract_answer,
    format_error,
    loop_repetition_ratio,
    process_reward,
    verify_process,
)

EXPR = "(221 + 427) * 28 - 8485"
GOLD = "9659"
GOOD_TRACE = """Let's calculate carefully.
221 + 427 = 648
648 * 28 = 18144
18144 - 8485 = 9659
<answer>9659</answer>"""


class TestAnswerExtraction(unittest.TestCase):
    def test_answer_tag(self):
        self.assertEqual(answer_score("<answer>9659</answer>", GOLD), 1.0)
        self.assertFalse(format_error("<answer>9659</answer>"))

    def test_bare_numeric_final_line_is_scored_but_flagged(self):
        self.assertEqual(answer_score("...\n9659", GOLD), 1.0)
        self.assertTrue(format_error("...\n9659"))

    def test_multiple_tags_are_not_rescued(self):
        text = "<answer>1</answer><answer>9659</answer>"
        self.assertIsNone(extract_answer(text))
        self.assertEqual(answer_score(text, GOLD), 0.0)

    def test_prose_final_is_not_scraped(self):
        self.assertEqual(answer_score("The final answer is 9659.", GOLD), 0.0)


class TestExpressionParsing(unittest.TestCase):
    def test_symbolic_prompt(self):
        prompt = f"Compute exactly: {EXPR}. Return the final integer in <answer>...</answer>."
        self.assertEqual(expression_from_prompt(prompt), EXPR)

    def test_word_problem_prompt(self):
        prompt = (
            "A factory starts with 723 units, produces 319 units in each of 12 batches, "
            "then removes 7699 defective units. How many remain? Return the final integer "
            "in <answer>...</answer>."
        )
        expression = expression_from_prompt(prompt)
        self.assertEqual(expression, "723 + 319 * 12 - 7699")
        self.assertEqual(canonical_derivation(expression)["value"], -3148)

    def test_canonical_intermediates(self):
        info = canonical_derivation(EXPR)
        self.assertEqual(info["value"], 9659)
        self.assertEqual(info["intermediate_values"], [648, 18144])


class TestProcessVerification(unittest.TestCase):
    def test_full_trace(self):
        metrics = verify_process(EXPR, GOOD_TRACE, GOLD)
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)
        self.assertEqual(metrics["process_valid_steps"], 3)
        self.assertEqual(metrics["process_invalid_steps"], 0)

    def test_trivial_equation_spam_cannot_earn_process_reward(self):
        spam = "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n<answer>9659</answer>"
        metrics = verify_process(EXPR, spam, GOLD)
        self.assertEqual(metrics["process_coverage"], 0.0)
        self.assertEqual(metrics["process_valid_steps"], 0)
        reward = process_reward(None, spam, GOLD, expression=EXPR)
        # Only the final answer and format bonus survive.
        self.assertAlmostEqual(reward["score"], 1.1)

    def test_correct_intermediates_with_wrong_final_get_dense_credit(self):
        trace = "\n".join(
            [
                "221 + 427 = 648",
                "648 * 28 = 18144",
                "18144 - 8485 = 9600",
                "<answer>9600</answer>",
            ]
        )
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)
        self.assertEqual(metrics["process_invalid_steps"], 1)
        reward = process_reward(None, trace, GOLD, expression=EXPR)
        self.assertGreater(reward["score"], 0.5)
        self.assertEqual(reward["acc"], 0.0)

    def test_ungrounded_equations_are_rejected(self):
        # 300 + 300 = 600 matches a gold intermediate but neither operand is
        # grounded in the prompt or a previously accepted step.
        trace = "300 + 300 = 600\n600 + 100 = 700\n700 - 50 = 650\n<answer>650</answer>"
        metrics = verify_process("100 + 200 * 3 - 50", trace, "650")
        self.assertEqual(metrics["process_coverage"], 0.0)
        self.assertEqual(metrics["process_ungrounded_steps"], 3)

    def test_grounded_but_irrelevant_steps_earn_no_coverage(self):
        # All leaves are prompt literals and the arithmetic is valid, but none
        # of the results is a gold intermediate value.
        trace = "100 + 200 = 300\n100 - 50 = 50\n<answer>650</answer>"
        metrics = verify_process("100 + 200 * 3 - 50", trace, "650")
        self.assertEqual(metrics["process_coverage"], 0.0)
        self.assertEqual(metrics["process_valid_steps"], 2)
        self.assertEqual(metrics["process_invalid_steps"], 0)

    def test_arithmetic_error_is_penalised(self):
        trace = "221 + 427 = 649\n<answer>9659</answer>"
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertEqual(metrics["process_valid_steps"], 0)
        self.assertEqual(metrics["process_invalid_steps"], 1)
        reward = process_reward(None, trace, GOLD, expression=EXPR)
        self.assertAlmostEqual(reward["score"], 1.1 - 0.05)

    def test_parenthesised_equations(self):
        trace = "(221 + 427) = 648\n(648 * 28) = 18144\n<answer>9659</answer>"
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)

    def test_latex_operators_and_offsets(self):
        trace = (
            "$$\n(221 + 427) \\times 28 - 8485\n$$\n"
            "221 + 427 = 648\n"
            "648 \\times 28 = 18144\n"
            "18144 - 8485 = 9659\n"
            "<answer>9659</answer>"
        )
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)
        self.assertEqual(metrics["process_valid_steps"], 3)
        self.assertTrue(all(0 < end <= len(trace) for end in metrics["verified_char_ends"]))
        # The second verified step ends just after the value 18144 in the
        # original (LaTeX) string, not in the normalised copy.
        second_end = metrics["verified_char_ends"][1]
        self.assertIn("18144", trace[max(0, second_end - 12) : second_end])

    def test_unicode_operators(self):
        trace = "221 + 427 = 648\n648 × 28 = 18144\n18144 − 8485 = 9659\n<answer>9659</answer>"
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)

    def test_word_problem_trace(self):
        expression = "723 + 319 * 12 - 7699"
        trace = (
            "319 * 12 = 3828\n723 + 3828 = 4551\n4551 - 7699 = -3148\n"
            "<answer>-3148</answer>"
        )
        metrics = verify_process(expression, trace, "-3148")
        self.assertAlmostEqual(metrics["process_coverage"], 1.0)
        self.assertAlmostEqual(answer_score(trace, "-3148"), 1.0)

    def test_loop_penalty_only_targets_degenerate_text(self):
        loop = "451 * 982 = 441, " * 50 + "<answer>0</answer>"
        self.assertEqual(loop_repetition_ratio(GOOD_TRACE), 0.0)
        self.assertGreater(loop_repetition_ratio(loop), 0.5)
        base = process_reward(None, loop, GOLD, expression=EXPR)
        penalised = process_reward(
            None, loop, GOLD, expression=EXPR, loop_penalty_weight=0.2
        )
        self.assertLess(penalised["score"], base["score"])
        self.assertGreater(penalised["loop_penalty"], 0.0)
        self.assertEqual(base["loop_penalty"], 0.0)

    def test_duplicate_steps_do_not_inflate_coverage(self):
        trace = "221 + 427 = 648\n221 + 427 = 648\n221 + 427 = 648\n<answer>9659</answer>"
        metrics = verify_process(EXPR, trace, GOLD)
        self.assertEqual(metrics["process_total_steps"], 1)
        self.assertAlmostEqual(metrics["process_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
