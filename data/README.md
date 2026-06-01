# Data

Result files from the experiments reported in the paper.

| File | Description |
|------|-------------|
| `api_validation_final_200.csv` | Per-trial final outcomes, 200-trial ChatGPT API validation (with `source_run`: pilot40 / run160). |
| `api_validation_rounds_200.csv` | Round-level transcripts for the 200 validation trials (agent action, counterpart reply, utilities). |
| `api_validation_summary_200.csv` | Summary metrics for the combined 200-trial validation. |
| `api_validation_summary_heldout160.csv` | Summary metrics for the 160-trial held-out validation set. |
| `offline_ablation_summary.csv` | Offline simulation ablation across the three architectures (2,000 eval episodes per role). |
| `baseline_stress_summary_45.csv` | Baseline (pure-LLM / prompt-only / rule-based) stress-test summary. |
| `fullmodel_stress_summary.csv` | Full proposed model stress-test summary by counterpart style. |

API runs used ChatGPT 5.5 (OpenAI Responses API, temperature 0.7, six-round deadline).
No API keys or private data are included.
