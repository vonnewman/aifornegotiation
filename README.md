# AI for Negotiation — Principal-Aligned LLM Negotiation Agent

Code accompanying the paper *Principal-Aligned Negotiation Mediated by Large
Language Models: A Partially Observable and Risk-Sensitive Reinforcement
Learning Framework for Multi-Issue Bargaining* (submitted to *Group Decision
and Negotiation*).

The framework separates an LLM dialogue layer from agreement authority. Strategic
decisions are governed by a partially observable Markov decision process (POMDP)
belief state, a Bayesian opponent model, a conditional value-at-risk (CVaR) safety
module, and a tabular Q-learning policy, over a five-issue procurement contract
(unit price, quantity, delivery time, payment term, contract duration).

## Repository layout

```
src/
  bayesian_safe_rl_negotiation_compare.py   Offline simulation + 3-architecture ablation
  five_issue_negotiation_sim.py             Five-issue offline negotiation simulator
  pomdp_cvar_openai_api_experiment.py       200-trial ChatGPT API validation (proposed model)
  gdn_api_baseline_stress_experiment.py     Baseline + stress-test API experiments
  five_issue_openai_api_experiment.py       Five-issue API experiment driver
```

## Requirements

Python 3.10+. The offline simulation uses only the standard library. The API
experiments use `urllib` from the standard library and require an OpenAI API key.

## Reproducing the offline results (no API key)

```bash
python src/bayesian_safe_rl_negotiation_compare.py
```

This trains and evaluates the three architectures (Principal-first Q-learning,
Bayesian Safe Q-learning, full risk-sensitive POMDP+CVaR model) for both buyer
and seller roles and writes summary CSVs. The run is deterministic given the
seed in the script.

## Reproducing the API results (requires an OpenAI API key)

```bash
export OPENAI_API_KEY="<your key>"
export OPENAI_MODEL="gpt-5.5"   # model used in the paper (ChatGPT 5.5)
python src/pomdp_cvar_openai_api_experiment.py
python src/gdn_api_baseline_stress_experiment.py
```

API runs are not bit-for-bit reproducible because the model is stochastic
(temperature 0.7) and non-stationary across versions.

## Notes

- Issue ranges, weights, and role mandates are defined in
  `bayesian_safe_rl_negotiation_compare.py` and imported by the API scripts.
- No API keys or private data are stored in this repository.

## Data

Result CSVs are in `data/` (see `data/README.md`), including the 200-trial API
validation (per-trial and round-level), the held-out 160-trial summary, the
offline ablation, and the baseline/stress-test summaries.

## License

MIT License (see `LICENSE`).
