---
description: Log the most recent experiment to docs/research-log.md
---

Review the most recent config file used, the results directory for that experiment, and any recent code changes. Then:

1. Generate the experiment ID from the config's `experiment_id` field (format: exp_YYYYMMDD_shortname)
2. Append a complete entry to docs/research-log.md:

### {experiment_id}
**Date:** YYYY-MM-DD
**Goal:** What we were testing
**Method:** What we did (brief — model, dataset, sweep params)
**Config:** configs/{experiment_id}.yaml
**Results:** Key numbers (best layer, best coef, accuracy delta, mean_delta_sum_logprob, etc.)
**Plots:** List paths to generated plots in results/{experiment_id}/
**Verdict:** What this means / what to try next
**Status:** [success / partial / failed / inconclusive]

3. Add a one-line entry to docs/experiment-index.md (| id | date | goal | status | key result |)
4. If the result meaningfully changes our understanding (new best layer, surprising coef behavior, etc.), add or update an entry in docs/findings.md
5. Commit the doc changes with message: "docs: log {experiment_id}"
