"""Zero-shot validity prediction with a 4-bit open model (fits a free Colab T4).

Instead of parsing free text, we read the model's next-token scores for
"Yes" vs "No" right after the question. This never fails to parse and gives a
probability we can use later (e.g., for thresholds or calibration).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ZERO_SHOT_PROMPT = (
    "Decide whether the following argument is logically valid. "
    "An argument is valid if the conclusion must follow from the premises, "
    "whether or not the premises or conclusion are true in the real world.\n\n"
    "Argument: {syllogism}\n\n"
    "Is the argument logically valid? Answer with Yes or No only."
)


def load_model(model_name):
    """Load a causal LM in 4-bit. T4 GPUs have no bfloat16, so compute in float16."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # so the last position is the real last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    return tok, model


def _answer_token_ids(tok, words):
    """First token id of each spelling variant (deduplicated)."""
    ids = set()
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def build_prompt(tok, syllogism, model_name, template=ZERO_SHOT_PROMPT):
    messages = [{"role": "user", "content": template.format(syllogism=syllogism)}]
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs["enable_thinking"] = False  # direct answer, no long reasoning
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


@torch.no_grad()
def predict_validity(items, tok, model, model_name, template=ZERO_SHOT_PROMPT, batch_size=8):
    """Return one prediction per item: {id, validity, p_yes}."""
    yes_ids = _answer_token_ids(tok, ["Yes", "yes", " Yes", " yes"])
    no_ids = _answer_token_ids(tok, ["No", "no", " No", " no"])
    preds = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [build_prompt(tok, x["syllogism"], model_name, template) for x in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        lp_yes = torch.logsumexp(logp[:, yes_ids], dim=-1)
        lp_no = torch.logsumexp(logp[:, no_ids], dim=-1)
        p_yes = torch.sigmoid(lp_yes - lp_no)  # P(Yes) renormalised over Yes/No
        for x, p in zip(batch, p_yes.tolist()):
            preds.append({"id": x["id"], "validity": bool(p >= 0.5), "p_yes": round(p, 4)})
    return preds
