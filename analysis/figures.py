"""Figures for the report. Usage: python figures.py <out_dir with scores_with_ci.csv and distractor_change.csv>"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
BLUE, ORANGE = "#2a78d6", "#eb6834"           # categorical slots 1 and 2 (validated, CVD-safe)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_NAMES = {"llama3.2-3b": "Llama 3.2 3B", "llama3.1-8b": "Llama 3.1 8B", "qwen3-8b": "Qwen3 8B"}
METHOD_NAMES = {"zeroshot": "Zero-shot", "cot": "Chain-of-thought", "abstract": "Abstraction"}
METHODS = ["zeroshot", "cot", "abstract"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})

rows = list(csv.DictReader(open(OUT / "scores_with_ci.csv")))
get = lambda st, model, method: next(r for r in rows if r["subtask"] == str(st) and r["split"] == "test"
                                     and r["model"] == model and r["method"] == method)


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


# ---------- Figure 1: "says valid" when the conclusion sounds true vs. false (Subtask 1 test) ----------
models = ["llama3.2-3b", "llama3.1-8b", "qwen3-8b"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, sharey=True)
for ax, model in zip(axes, models):
    style(ax)
    for y, method in enumerate(reversed(METHODS)):
        r = get(1, model, method)
        p, q = float(r["say_valid_if_plausible"]), float(r["say_valid_if_implausible"])
        ax.plot([q, p], [y, y], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
        # square drawn first and larger, circle on top: when the two coincide both stay visible
        ax.scatter([q], [y], s=58, color=ORANGE, marker="s", edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([p], [y], s=30, color=BLUE, marker="o", edgecolor="white", linewidth=1.2, zorder=4)
        gap = p - q
        ax.text(103, y, f"{gap:+.0f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_title(MODEL_NAMES[model], fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(range(3))
    ax.set_yticklabels([METHOD_NAMES[m] for m in reversed(METHODS)])
    ax.axvline(50, color=GRID, linewidth=0.8, zorder=0)
axes[1].set_xlabel('% of items the model calls "valid"')
for ax in axes:
    ax.text(103, 2.55, "gap", ha="left", fontsize=7.5, color=INK2)
handles = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=5.5, label="Conclusion sounds true"),
           plt.Line2D([], [], marker="s", color=ORANGE, linestyle="", markersize=7, label="Conclusion sounds false")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 0.98, 0.95))
fig.savefig(OUT / "fig1_says_valid_gap.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig1_says_valid_gap.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Figure 2: TCE without vs. with irrelevant premises, 95% CI ----------
d_rows = list(csv.DictReader(open(OUT / "distractor_change.csv")))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharex=True, sharey=True)
for ax, model in zip(axes, ["llama3.1-8b", "qwen3-8b"]):
    style(ax)
    for y, method in enumerate(reversed(METHODS)):
        for st, color, marker, dy in [(1, BLUE, "o", 0.13), (2, ORANGE, "s", -0.13)]:
            r = get(st, model, method)
            v, lo, hi = float(r["tce"]), float(r["tce_lo"]), float(r["tce_hi"])
            ax.plot([lo, hi], [y + dy, y + dy], color=color, linewidth=1.6, alpha=0.55, solid_capstyle="round")
            ax.scatter([v], [y + dy], s=36, color=color, marker=marker, edgecolor="white", linewidth=1, zorder=3)
        d = next(x for x in d_rows if x["model"] == model and x["method"] == method)
        sig = float(d["change_tce_lo"]) > 0 or float(d["change_tce_hi"]) < 0
        ax.text(76, y, f"{float(d['change_tce']):+.1f}" + (" *" if sig else ""), va="center", fontsize=8.5,
                color=INK if sig else INK2, fontweight="bold" if sig else "normal")
    ax.set_title(MODEL_NAMES[model], fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xlim(0, 72)
    ax.set_yticks(range(3))
    ax.set_yticklabels([METHOD_NAMES[m] for m in reversed(METHODS)])
    ax.set_xlabel("Total content effect (TCE), lower is better")
    ax.text(76, 2.55, "change", fontsize=7.5, color=INK2)
handles = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="-", markersize=6, label="Subtask 1 (no distractors)"),
           plt.Line2D([], [], marker="s", color=ORANGE, linestyle="-", markersize=5.5, label="Subtask 2 (irrelevant premises)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 0.97, 0.94))
fig.savefig(OUT / "fig2_distractor_tce.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig2_distractor_tce.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("figures written")
