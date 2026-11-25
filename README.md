# Content Effects in LLM Syllogistic Reasoning (SemEval-2026 Task 11)

CS 421 (NLP) course project, University of Illinois Chicago.

The task: decide whether a syllogism is **logically valid**, regardless of whether its conclusion sounds true. This project measures how strongly three open LLMs are swayed by believable content (the *content effect*), tests two low-cost prompting fixes, and checks what irrelevant premises do.

**Main finding:** the right fix depends on the model. Abstraction (rewriting terms as letters) removes most of the content effect for Llama models; chain-of-thought works best for Qwen3.

| Best system | Subtask | Accuracy | TCE (lower is better) | CS |
|---|---|---|---|---|
| Qwen3 8B + chain-of-thought | 1 | 84.3 | 6.3 | 28.28 |
| Llama 3.1 8B + abstraction | 2 | 61.1 (premise F1 92.1) | 3.5 | 30.58 |

Full write-up: [`report/report.pdf`](report/CS421Task11.pdf).

## Repository layout

```
src/
  data.py        load official data, remove 2 train/test overlaps, balanced 160-item dev split
  model.py       4-bit model loading; prompt building; Yes/No next-token scoring (zero-shot)
  prompting.py   chain-of-thought and abstraction methods
  subtask2.py    premise splitting and premise selection for Subtask 2
  scoring.py     official metric (Acc, TCE, CS, premise F1) using the official functions
notebooks/       Colab notebook that ran every experiment
data/            dev.json, train_split.json
results/         st<subtask>_<split>_<model>_<method>_{preds,scores}.json for every run
analysis/        bootstrap confidence intervals, paired tests, figures (no GPU needed)
report/          final report (PDF and LaTeX)
```

## Setup and how to run

1. Open the notebook in Google Colab with a **T4 GPU** runtime.
2. Install: `pip install -r requirements.txt`, then restart the session.
3. Add a Hugging Face token to Colab Secrets (Llama models are gated and need approved access).
4. The official data and scorer are cloned automatically into `official/` at commit `6f71ea8` from
   https://github.com/neuro-symbolic-ai/semeval_2026_task_11 (not included here).
5. Run the notebook top to bottom. Finished runs are skipped, so it can resume after a disconnect.
6. Analysis (CPU only):
   ```
   python analysis/analysis.py results official analysis/output
   python analysis/figures.py analysis/output
   ```

## Models

`meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B` (thinking mode off), all in 4-bit NF4 with float16 compute.

## Notes

- Public test files have 191 (Subtask 1) and 190 (Subtask 2) items; the overview paper reports 192.
- Prompts were fixed before running the test set and were not tuned.
