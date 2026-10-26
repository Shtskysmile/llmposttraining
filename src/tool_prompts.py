"""Shared prompts for direct and Python-tool math rollouts."""


TOOL_SYSTEM = (
    "Please integrate natural language reasoning with programs to solve the problem above, and put your "
    "final answer within \\boxed{}. You may call the Python calculator using a fenced ```python code block. "
    "The calculator permits arithmetic, combinations, permutations, product, comb, gcd, and lcm from "
    "Python builtins, itertools, and math; do not use sympy. After the tool returns, use its result and "
    "provide the exact final value in <answer>...</answer>."
)


TOOL_DEMONSTRATIONS = [
    ("From 1 through 8, choose 2 numbers, no two consecutive, with sum congruent to 1 modulo 3. How many choices?",
     "```python\nchoices = [xs for xs in combinations(range(1, 9), 2) if xs[1] - xs[0] > 1 and sum(xs) % 3 == 1]\nprint(len(choices))\n```", "7"),
    ("A bag has 3 red and 2 blue balls. Two are drawn without replacement. Given at least one is red, what is the probability exactly one is red?",
     "```python\nfavorable = 3 * 2\nconditioning = comb(5, 2) - comb(2, 2)\ng = gcd(favorable, conditioning)\nprint(f'{favorable // g}/{conditioning // g}')\n```", "2/3"),
    ("Find the least nonnegative x with x congruent to 2 modulo 3 and x congruent to 3 modulo 5.",
     "```python\nsolutions = [x for x in range(lcm(3, 5)) if x % 3 == 2 and x % 5 == 3]\nprint(min(solutions))\n```", "8"),
]


def tool_messages(prompt: str, few_shot: bool = True):
    """Build the same tool-enabled context for training and evaluation."""
    messages = [{"role": "system", "content": TOOL_SYSTEM}]
    if few_shot:
        for question, code, result in TOOL_DEMONSTRATIONS:
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Tool result (success=True):\n{result}\nGive the exact result in <answer>...</answer>."},
                {"role": "assistant", "content": f"<answer>{result}</answer>"},
            ])
    messages.append({"role": "user", "content": prompt + "\nPut the exact final value in <answer>...</answer>."})
    return messages
