"""Chain-of-thought (CoT) and abstraction prompting.

Both methods let the model write text first, then read its final answer the
same way as zero-shot: compare the next-token scores for "Yes" vs "No".
- CoT:         model reasons step by step -> append "Final answer (Yes or No):" -> score Yes/No.
- Abstraction: model rewrites the argument with letters (A, B, C) -> the judge sees ONLY that version.
"""
import torch
from model import ZERO_SHOT_PROMPT, _answer_token_ids, build_prompt

COT_PROMPT = (
    "Decide whether the following argument is logically valid. "
    "An argument is valid if the conclusion must follow from the premises, "
    "whether or not the premises or conclusion are true in the real world.\n\n"
    "Argument: {syllogism}\n\n"
    "Think step by step about the logical structure only, ignoring whether the "
    "statements are true in reality. Keep your reasoning short. "
    "End with your final answer: Yes (valid) or No (invalid)."
)

ABSTRACT_PROMPT = (
    "Rewrite the following argument in abstract form. Replace each category or "
    "term with a capital letter (A, B, C), using the same letter every time the "
    "same term appears. Keep the logical words (all, some, no, not, therefore) "
    "and the order of the sentences. Output only the rewritten argument, nothing else.\n\n"
    "Argument: {syllogism}"
)

COT_ANSWER_CUE = "\n\nFinal answer (Yes or No):"


@torch.no_grad()
def generate(prompts, tok, model, max_new_tokens=256, batch_size=8):
    """Greedy generation for already-templated prompts. Returns the new text only."""
    outs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


@torch.no_grad()
def p_yes_for_prompts(prompts, tok, model, batch_size=8):
    """P(Yes) renormalised over Yes/No, read at the last position of each prompt."""
    yes_ids = _answer_token_ids(tok, ["Yes", "yes", " Yes", " yes"])
    no_ids = _answer_token_ids(tok, ["No", "no", " No", " no"])
    probs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), dim=-1)
        diff = torch.logsumexp(logp[:, yes_ids], dim=-1) - torch.logsumexp(logp[:, no_ids], dim=-1)
        probs += torch.sigmoid(diff).tolist()
    return probs


def predict_cot(items, tok, model, model_name, max_new_tokens=256, batch_size=8):
    prompts = [build_prompt(tok, x["syllogism"], model_name, COT_PROMPT) for x in items]
    reasoning = generate(prompts, tok, model, max_new_tokens, batch_size)
    probs = p_yes_for_prompts([p + r + COT_ANSWER_CUE for p, r in zip(prompts, reasoning)],
                              tok, model, batch_size)
    return [{"id": x["id"], "validity": bool(p >= 0.5), "p_yes": round(p, 4), "reasoning": r}
            for x, p, r in zip(items, probs, reasoning)]


def predict_abstract(items, tok, model, model_name, max_new_tokens=128, batch_size=8):
    prompts = [build_prompt(tok, x["syllogism"], model_name, ABSTRACT_PROMPT) for x in items]
    abstract = generate(prompts, tok, model, max_new_tokens, batch_size)
    probs = p_yes_for_prompts([build_prompt(tok, a, model_name, ZERO_SHOT_PROMPT) for a in abstract],
                              tok, model, batch_size)
    return [{"id": x["id"], "validity": bool(p >= 0.5), "p_yes": round(p, 4), "abstract": a}
            for x, p, a in zip(items, probs, abstract)]


METHODS = {"cot": predict_cot, "abstract": predict_abstract}
