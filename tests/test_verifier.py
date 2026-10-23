import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier import extract_answer, format_error, score


class VerifierTests(unittest.TestCase):
    def test_exact_integer(self):
        self.assertEqual(score("<answer>42</answer>", "42"), 1.0)

    def test_bare_final_number_is_scored_but_format_is_reported(self):
        self.assertEqual(score("42", "42"), 1.0)
        self.assertTrue(format_error("42"))

    def test_standalone_correct_equation_is_verifiable(self):
        self.assertEqual(score("84 × 13 = 1092", "1092"), 1.0)
        self.assertTrue(format_error("84 × 13 = 1092"))
        self.assertEqual(score("84 × 13 = 1093", "1093"), 0.0)

    def test_explicit_boxed_final_answer(self):
        self.assertEqual(score("work ... Final answer: $\\boxed{390}$", "390"), 1.0)
        self.assertTrue(format_error("work ... Final answer: $\\boxed{390}$"))
        self.assertEqual(score("Final answer:\n**5272**", "5272"), 1.0)
        self.assertEqual(score("Final answer: **5272**", "5272"), 0.0)

    def test_rational_numeric_normalization(self):
        self.assertEqual(score("Reasoning... <answer>6.0</answer>", "6"), 1.0)
        self.assertEqual(score("<answer>3/2</answer>", "1.5"), 1.0)

    def test_explanation_does_not_get_reward(self):
        self.assertEqual(score("The answer is 42.", "42"), 0.0)
        self.assertEqual(score("Answer: 42", "42"), 0.0)
        self.assertEqual(score("<answer>41</answer> The answer should be 42", "42"), 0.0)

    def test_malformed_or_multiple_tags_fail(self):
        for text in ("<answer>forty two</answer>", "<answer>42", "<answer>42</answer><answer>42</answer>", "<answer>1/0</answer>"):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))
                self.assertTrue(format_error(text))
                self.assertEqual(score(text, "42"), 0.0)

    def test_wrong_answer_is_not_format_error(self):
        self.assertFalse(format_error("<answer>41</answer>"))
        self.assertEqual(score("<answer>41</answer>", "42"), 0.0)


if __name__ == "__main__":
    unittest.main()
