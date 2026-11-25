"""Analysis for SemEval-2026 Task 11 project (no GPU needed).

Reads saved predictions from results/ and gold labels from the official repo, then computes:
  1. Point scores with the official metric functions (Acc, 4 group accuracies, TCE, CS; + premise F1 for Subtask 2)
  2. 95% bootstrap confidence intervals (items resampled with replacement, 2000 times)
  3. Paired bootstrap tests: each method vs. zero-shot, same model, same items
  4. "Says valid" rate when the conclusion sounds true vs. false
  5. Abstraction check: does the rewrite use exactly 3 letters (a syllogism has 3 terms)?
  6. CoT check: does the reasoning explicitly appeal to real-world truth?
  7. Distractor change: TCE and accuracy from Subtask 1 to Subtask 2 (different items, independent bootstrap)

Usage: python analysis.py <results_dir> <official_repo_dir> <out_dir>
"""
import csv
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

RESULTS, OFFICIAL, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
B, SEED = 2000, 0

MODELS = ["llama3.2-3b", "llama3.1-8b", "qwen3-8b"]
METHODS = ["zeroshot", "cot", "abstract"]


def load(p):
    with open(p) as f:
        return json.load(f)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev1 = load_module(OFFICIAL / "evaluation_kit/task 1 & 3/evaluation_script.py", "ev1")
ev2 = load_module(OFFICIAL / "evaluation_kit/task 2 & 4/evaluation_script.py", "ev2")

gold = {x["id"]: x for x in
        load(OFFICIAL / "train_data/subtask 1/train_data.json")
        + load(OFFICIAL / "test_data/subtask 1/test_data_subtask_1.json")
        + load(OFFICIAL / "test_data/subtask 2/test_data_subtask_2.json")}


# ---------- metric on a list of (gold, pred) pairs, using the official functions ----------
def metrics(pairs, subtask):
    g = [a for a, _ in pairs]
    p = [b for _, b in pairs]
    # official functions key by id; resampled items repeat ids, so give each copy a unique id
    g = [dict(x, id=f"{x['id']}#{i}") for i, x in enumerate(g)]
    p = [dict(x, id=f"{x['id']}#{i}") for i, x in enumerate(p)]
    gt = {x["id"]: x for x in g}
    acc = ev1.calculate_accuracy(g, p, "validity", "validity", None)[0]
    cells = {
        "acc_plausible_valid": ev1.calculate_subgroup_accuracy(gt, p, True, True)[0],
        "acc_implausible_valid": ev1.calculate_subgroup_accuracy(gt, p, True, False)[0],
        "acc_plausible_invalid": ev1.calculate_subgroup_accuracy(gt, p, False, True)[0],
        "acc_implausible_invalid": ev1.calculate_subgroup_accuracy(gt, p, False, False)[0],
    }
    tce = ev1.calculate_content_effect_bias(cells)["tot_content_effect"]
    out = {"accuracy": acc, "tce": tce, **cells}
    if subtask == 2:
        f1 = ev2.calculate_f1_premises(gt, p)
        out["f1_premises"] = f1
        out["cs"] = ev2.calculate_smooth_combined_metric((acc + f1) / 2, tce)
    else:
        out["cs"] = ev1.calculate_smooth_combined_metric(acc, tce)
    # how often the model says "valid", by plausibility of the conclusion
    for plaus, key in [(True, "say_valid_if_plausible"), (False, "say_valid_if_implausible")]:
        sel = [b for a, b in zip(g, p) if a["plausibility"] == plaus]
        out[key] = 100 * sum(b["validity"] for b in sel) / len(sel)
    return out


def ci(values):
    v = sorted(values)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def runs():
    for f in sorted(RESULTS.glob("st*_*_*_*_preds.json")):
        parts = f.name.replace("_preds.json", "").split("_")
        if len(parts) != 4 or parts[1] == "dev10":
            continue
        task, split, model, method = parts
        if method not in METHODS:
            continue
        yield int(task[2]), split, model, method, load(f)


# ---------- 1-3. scores, CIs, paired tests ----------
rng = random.Random(SEED)
rows, boot = [], {}
for subtask, split, model, method, preds in runs():
    pairs = [(gold[p["id"]], p) for p in preds]
    point = metrics(pairs, subtask)
    idx_sets = [[rng.randrange(len(pairs)) for _ in pairs] for _ in range(B)]
    samples = [metrics([pairs[i] for i in idx], subtask) for idx in idx_sets]
    key = (subtask, split, model, method)
    boot[key] = (pairs, idx_sets, samples)
    row = {"subtask": subtask, "split": split, "model": model, "method": method, "n": len(pairs)}
    for m in ["accuracy", "tce", "cs", "f1_premises", "say_valid_if_plausible", "say_valid_if_implausible",
              "acc_plausible_valid", "acc_implausible_valid", "acc_plausible_invalid", "acc_implausible_invalid"]:
        if m in point:
            row[m] = round(point[m], 2)
            if m in ("accuracy", "tce", "cs"):
                lo, hi = ci([s[m] for s in samples])
                row[m + "_lo"], row[m + "_hi"] = round(lo, 2), round(hi, 2)
    rows.append(row)

with open(OUT / "scores_with_ci.csv", "w", newline="") as f:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# paired: same resampled items for both methods (items are identical across methods of one model/split)
paired = []
for (subtask, split, model, method), (pairs, _, _) in boot.items():
    if method == "zeroshot" or (subtask, split, model, "zeroshot") not in boot:
        continue
    base_pairs = boot[(subtask, split, model, "zeroshot")][0]
    by_id = {g["id"]: (g, p) for g, p in base_pairs}
    base_aligned = [by_id[g["id"]] for g, _ in pairs]
    prng = random.Random(SEED + 1)
    d_acc, d_tce = [], []
    for _ in range(B):
        idx = [prng.randrange(len(pairs)) for _ in pairs]
        a = metrics([pairs[i] for i in idx], subtask)
        b = metrics([base_aligned[i] for i in idx], subtask)
        d_acc.append(a["accuracy"] - b["accuracy"])
        d_tce.append(a["tce"] - b["tce"])
    pa, pb = metrics(pairs, subtask), metrics(base_aligned, subtask)
    def p_two_sided(d):
        return round(min(1.0, 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / len(d)), 3)
    paired.append({"subtask": subtask, "split": split, "model": model, "method_vs_zeroshot": method,
                   "d_accuracy": round(pa["accuracy"] - pb["accuracy"], 2),
                   "d_accuracy_lo": round(ci(d_acc)[0], 2), "d_accuracy_hi": round(ci(d_acc)[1], 2),
                   "p_accuracy": p_two_sided(d_acc),
                   "d_tce": round(pa["tce"] - pb["tce"], 2),
                   "d_tce_lo": round(ci(d_tce)[0], 2), "d_tce_hi": round(ci(d_tce)[1], 2),
                   "p_tce": p_two_sided(d_tce)})

with open(OUT / "paired_tests_vs_zeroshot.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0]))
    w.writeheader()
    w.writerows(paired)

# ---------- 5. abstraction letter check ----------
LETTER = re.compile(r"\b([A-Z])\b")
abs_rows = []
for subtask, split, model, method, preds in runs():
    if method != "abstract" or split == "dev":
        continue
    n_terms_expected = 3 if subtask == 1 else None
    ok, bad = [], []
    for p in preds:
        letters = set(LETTER.findall(p["abstract"])) - {"I"}
        correct = gold[p["id"]]["validity"] == p["validity"]
        if subtask == 1:
            (ok if len(letters) == 3 else bad).append(correct)
        else:
            ok.append(correct)  # Subtask 2 has a varying number of terms; report distribution only
    abs_rows.append({"subtask": subtask, "model": model, "n": len(preds),
                     "rewrites_with_3_letters_pct": round(100 * len(ok) / len(preds), 1) if subtask == 1 else "",
                     "acc_when_3_letters": round(100 * sum(ok) / len(ok), 1) if subtask == 1 and ok else "",
                     "acc_otherwise": round(100 * sum(bad) / len(bad), 1) if bad else "",
                     "n_otherwise": len(bad) if subtask == 1 else ""})
with open(OUT / "abstraction_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(abs_rows[0]))
    w.writeheader()
    w.writerows(abs_rows)

# ---------- 6. CoT real-world appeal check ----------
# strict markers only: looser words ("we know that", "is true") are ordinary logical language
WORLD = re.compile(r"\b(in reality|real world|real life|in real-world|common knowledge|biologically|scientifically|in the real world|realistically)\b", re.I)
cot_rows = []
for subtask, split, model, method, preds in runs():
    if method != "cot" or split == "dev":
        continue
    w_, nw = [], []
    for p in preds:
        correct = gold[p["id"]]["validity"] == p["validity"]
        (w_ if WORLD.search(p["reasoning"]) else nw).append(correct)
    cot_rows.append({"subtask": subtask, "model": model, "n": len(preds),
                     "appeals_to_world_pct": round(100 * len(w_) / len(preds), 1),
                     "acc_when_appeals": round(100 * sum(w_) / len(w_), 1) if w_ else "",
                     "acc_otherwise": round(100 * sum(nw) / len(nw), 1) if nw else ""})
with open(OUT / "cot_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cot_rows[0]))
    w.writeheader()
    w.writerows(cot_rows)

# ---------- 7. Subtask 1 -> Subtask 2 change (different item sets: independent resamples) ----------
dist_rows = []
for model in MODELS:
    for method in METHODS:
        k1, k2 = (1, "test", model, method), (2, "test", model, method)
        if k1 not in boot or k2 not in boot:
            continue
        s1, s2 = boot[k1][2], boot[k2][2]
        p1, p2 = metrics(boot[k1][0], 1), metrics(boot[k2][0], 2)
        row = {"model": model, "method": method}
        for m in ["accuracy", "tce"]:
            d = [b[m] - a[m] for a, b in zip(s1, s2)]
            row[f"st1_{m}"], row[f"st2_{m}"] = round(p1[m], 2), round(p2[m], 2)
            row[f"change_{m}"] = round(p2[m] - p1[m], 2)
            row[f"change_{m}_lo"], row[f"change_{m}_hi"] = (round(v, 2) for v in ci(d))
        dist_rows.append(row)
with open(OUT / "distractor_change.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dist_rows[0]))
    w.writeheader()
    w.writerows(dist_rows)

print("Wrote:", ", ".join(p.name for p in sorted(OUT.iterdir())))
