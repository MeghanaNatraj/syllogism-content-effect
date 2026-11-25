"""Subtask 2: syllogisms with irrelevant premises.
1. Validity: same methods as Subtask 1, run on the full text including the irrelevant premises.
2. Premise selection: the model sees numbered premises and names the ones needed (official premise F1).
"""
import re
from model import build_prompt
from prompting import generate

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

PREMISE_PROMPT = (
    "Below are numbered premises and a conclusion. Most premises are irrelevant. "
    "Which premises are needed to logically derive the conclusion? Usually exactly two are needed.\n\n"
    "Premises:\n{premises}\n\n"
    "Conclusion: {conclusion}\n\n"
    "Answer with the numbers of the needed premises only, separated by commas (for example: 2, 5)."
)

# The model's reply is started for it, so it answers with numbers straight away
# instead of explaining first (Qwen3 otherwise explains and gets cut off).
ANSWER_PREFIX = "Needed premises:"


def split_premises(text):
    """Return (list of premise sentences, conclusion). The conclusion is the last sentence."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return sents[:-1], sents[-1]


def _parse_numbers(answer, n_premises):
    """Turn '2, 5' (1-based) into [1, 4] (0-based). Reads only the first line; ignores numbers out of range."""
    first_line = answer.strip().split("\n")[0]
    nums = [int(n) for n in re.findall(r"\d+", first_line)]
    return sorted({n - 1 for n in nums if 1 <= n <= n_premises})


def predict_premises(items, tok, model, model_name, batch_size=8):
    prompts, sizes = [], []
    for x in items:
        premises, conclusion = split_premises(x["syllogism"])
        numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises))
        text = PREMISE_PROMPT.format(premises=numbered, conclusion=conclusion)
        prompts.append(build_prompt(tok, text, model_name, template="{syllogism}") + ANSWER_PREFIX)
        sizes.append(len(premises))
    answers = generate(prompts, tok, model, max_new_tokens=12, batch_size=batch_size)
    return [{"id": x["id"], "relevant_premises": _parse_numbers(a, n), "premise_answer": a}
            for x, a, n in zip(items, answers, sizes)]
