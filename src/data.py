"""Load SemEval-2026 Task 11 data and build the train/dev split.

The official repo is cloned (pinned commit) into OFFICIAL_DIR. Nothing here
touches the test labels except to find train items that also appear in test.
"""
import json
import random
import subprocess
from pathlib import Path

OFFICIAL_REPO = "https://github.com/neuro-symbolic-ai/semeval_2026_task_11.git"
OFFICIAL_COMMIT = "6f71ea846770f9418c7d2845ec6c25e59f885bd7"  # Mar 2, 2026

CELLS = [(True, True), (True, False), (False, True), (False, False)]  # (validity, plausibility)


def clone_official(official_dir="official"):
    """Clone the official repo at the pinned commit (skipped if already present)."""
    official_dir = Path(official_dir)
    if not (official_dir / ".git").exists():
        subprocess.run(["git", "clone", "-q", OFFICIAL_REPO, str(official_dir)], check=True)
    subprocess.run(["git", "-C", str(official_dir), "checkout", "-q", OFFICIAL_COMMIT], check=True)
    return official_dir


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_official(official_dir="official"):
    """Return (train, test_subtask1, test_subtask2) as lists of dicts."""
    d = Path(official_dir)
    train = load_json(d / "train_data" / "subtask 1" / "train_data.json")
    test1 = load_json(d / "test_data" / "subtask 1" / "test_data_subtask_1.json")
    test2 = load_json(d / "test_data" / "subtask 2" / "test_data_subtask_2.json")
    return train, test1, test2


def cell_of(item):
    return (item["validity"], item["plausibility"])


def make_splits(train, tests, dev_per_cell=40, seed=42):
    """Drop train items whose text also appears in any test set, then hold out
    `dev_per_cell` items from each validity x plausibility cell as the dev set.

    Returns (train_split, dev_split, removed_overlap).
    """
    test_texts = {x["syllogism"].strip() for t in tests for x in t}
    removed = [x for x in train if x["syllogism"].strip() in test_texts]
    pool = [x for x in train if x["syllogism"].strip() not in test_texts]

    rng = random.Random(seed)
    dev, train_split = [], []
    for cell in CELLS:
        items = [x for x in pool if cell_of(x) == cell]
        rng.shuffle(items)
        dev += items[:dev_per_cell]
        train_split += items[dev_per_cell:]
    rng.shuffle(dev)
    rng.shuffle(train_split)
    return train_split, dev, removed


def cell_counts(items):
    return {f"valid={v}, plausible={p}": sum(cell_of(x) == (v, p) for x in items) for v, p in CELLS}
