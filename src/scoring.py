"""Official Task 11 metrics, using the functions from the official evaluation scripts."""
import importlib.util
from pathlib import Path

_SCRIPT = Path("evaluation_kit") / "task 1 & 3" / "evaluation_script.py"
_SCRIPT_ST2 = Path("evaluation_kit") / "task 2 & 4" / "evaluation_script.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(gold, preds, official_dir="official"):
    """Subtask 1 metric: accuracy, 4 group accuracies, TCE and CS (all in %)."""
    ev = _load(Path(official_dir) / _SCRIPT, "official_eval_13")
    missing = {g["id"] for g in gold} - {p["id"] for p in preds}
    if missing:
        raise ValueError(f"{len(missing)} gold items have no prediction")
    if any(not isinstance(p.get("validity"), bool) for p in preds):
        raise ValueError("Some predictions have a non-boolean 'validity'")
    gt_map = {g["id"]: g for g in gold}
    acc, _, _ = ev.calculate_accuracy(gold, preds, "validity", "validity", None)
    cells = {
        "acc_plausible_valid": ev.calculate_subgroup_accuracy(gt_map, preds, True, True)[0],
        "acc_implausible_valid": ev.calculate_subgroup_accuracy(gt_map, preds, True, False)[0],
        "acc_plausible_invalid": ev.calculate_subgroup_accuracy(gt_map, preds, False, True)[0],
        "acc_implausible_invalid": ev.calculate_subgroup_accuracy(gt_map, preds, False, False)[0],
    }
    bias = ev.calculate_content_effect_bias(cells)
    cs = ev.calculate_smooth_combined_metric(acc, bias["tot_content_effect"])
    return {"n": len(gold), "accuracy": round(acc, 2),
            **{k: round(v, 2) for k, v in cells.items()},
            "tce_intra": round(bias["content_effect_intra_validity_label"], 2),
            "tce_cross": round(bias["content_effect_inter_validity_label"], 2),
            "tce": round(bias["tot_content_effect"], 2), "cs": round(cs, 2)}


def score_st2(gold, preds, official_dir="official"):
    """Subtask 2 metric: adds premise F1; CS = ((accuracy + F1) / 2) / (1 + ln(1 + TCE))."""
    r = score(gold, preds, official_dir)
    ev2 = _load(Path(official_dir) / _SCRIPT_ST2, "official_eval_24")
    f1 = ev2.calculate_f1_premises({g["id"]: g for g in gold}, preds)
    r["f1_premises"] = round(f1, 2)
    r["cs"] = round(ev2.calculate_smooth_combined_metric((r["accuracy"] + f1) / 2.0, r["tce"]), 2)
    return r
