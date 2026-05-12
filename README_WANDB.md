# Weights & Biases Integration

Reference for the W&B logging built into `experiments/runner.py`. Covers setup,
configuration, what gets logged, and the full parameter dictionary.

---

## What's Wired Up

- **`experiments/wandb_logger.py`** — soft-imports `wandb` so the project still
  runs without it installed. All helpers accept `run=None` to mean "logging
  disabled" so callers stay unconditional.
- **`runner.py`** — wraps the experiment body in `init_run` / `finish_run`
  (try/finally) and passes per-layer callbacks into `sweep_layers_mcf` and
  `sweep_layers_cf`. The CF target-only path logs inline by reading
  `combined_summary.csv` after each layer.

Key behaviors:

- `reinit=True` on `wandb.init` — sequential runs (`configs/*.yaml`) produce one
  W&B run per experiment.
- Step axis = **layer index**. Not wall-clock.
- Coefficients are encoded in the metric key, e.g. `mcf/coef_p5/accuracy`,
  `mcf/coef_n10/accuracy` (`+`/`-` get replaced with `p`/`n`). Each layer logs
  all coefs in a single `run.log()` call.
- A flat `layer_table` is also logged per layer for custom W&B panels.
- NaN/Inf values are filtered before logging.

---

## Setup From Scratch

### 1. Install

```bash
pip install wandb
```

### 2. Authenticate

```bash
wandb login
```

Paste your API key from https://wandb.ai/authorize. Writes `~/.netrc`. Once
per machine.

To skip auth entirely, set `wandb.mode: offline` in the config — runs write to
`./wandb/` locally and can be synced later with `wandb sync <run-dir>`.

### 3. Enable in a config

```yaml
wandb:
  enabled: true
  project: steering-vectors
  entity: null           # null = default account; otherwise team/user
  tags: [llama3-8b, anatomy, mcf-cf]
  notes: "Anatomy sweep, all 32 layers"
  mode: online           # online | offline | disabled
```

If `enabled: false` (the default) or `wandb` isn't installed, the runner logs a
warning and continues without any W&B calls.

### 4. Smoke test before a full sweep

```bash
python experiments/runner.py configs/exp_test_wandb.yaml
```

This config is MCF-only, 2 layers, `mode: offline`, generation disabled — runs
in a few minutes. You should see `[wandb] run initialised:` near the top and
a `./wandb/offline-run-*/` directory afterwards.

### 5. Run a real config

```bash
python experiments/runner.py configs/exp_20260505_gpqa_physics_llama8b.yaml
```

Find the run at `https://wandb.ai/<entity>/steering-vectors`, named after the
`experiment_id`.

---

## Config Reference

### `wandb` section

| Param | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. If `false`, the runner skips W&B entirely. |
| `project` | str | `steering-vectors` | W&B project name. |
| `entity` | str / null | `null` | Team or user. `null` = default account. |
| `tags` | list[str] | `[]` | Filterable tags. |
| `notes` | str | `""` | Free-text run description. |
| `mode` | str | `online` | `online` (sync live), `offline` (write to `./wandb/`, sync later), `disabled` (no-op). |

### Inheritance with `base:`

Child configs that use `base:` get `wandb` deep-merged from the parent. Override
only what differs — typically just `tags` and `notes`:

```yaml
base: configs/parent.yaml
experiment_id: exp_child

wandb:
  tags: [llama3-8b, anatomy, cf-only]
  notes: "CF-only variant"
```

`enabled`, `project`, `mode` inherit from the parent.

### Full parameter dictionary (everything else)

#### `model`
| Param | Default | Notes |
|---|---|---|
| `name` | — | HuggingFace model ID. |
| `dtype` | `float16` | `float16` \| `bfloat16` \| `float32`. |
| `device` | `cuda` | `cuda` or `cpu`. |

#### `dataset`
Two modes, chosen by whether `neg_capture_path` is set.

| Param | Default | Notes |
|---|---|---|
| `eval_path` | — | CSV with `prompt, target` (+ optional `false1..3` for CF). |
| `neg_capture_path` | `""` | If set → **dataset capture mode**. Uses this CSV's rows as negative DIM examples and few-shot source. |
| `capture_n` | 100 | Rows used per side for DIM capture (dataset mode). |
| `positive_prompts_path` | `""` | **Text-file mode**: one positive prompt per line. |
| `negative_prompts_path` | `""` | Same, negatives. |

**Capture modes:**
- *Text-file*: pos/neg are plain sentences from `.txt` files. Simplest.
- *Dataset capture*: pos/neg are MCF-formatted rows from CSV. Controls for
  surface-form confounds.

#### `sweep` — main knobs

**What to score**
| Param | Default | Notes |
|---|---|---|
| `formulation` | `both` | `mcf` (single-letter logprob), `cf` (full target logprob), or `both`. |
| `cf_normalization` | `character` | `none` \| `token` \| `character` \| `pmi`. |
| `task_prefix` | `question` | `question` \| `goal` \| `fill_in_the_blank` \| `continuation`. |

**Which layers / sublayers**
| Param | Default | Notes |
|---|---|---|
| `layers` | `null` | List of int indices or `null` = all. CLI `--layers 0-5 7` overrides. |
| `layer_name_pattern` | `"model.layers.{layer_idx}"` | Module path template. |
| `sublayer` | `null` | Appended to layer pattern. E.g. `mlp`, `mlp.down_proj`, `self_attn.o_proj`. `null` = full block output. |
| `token_position` | `last` | `last` or `mean` — where activations are captured / vector is injected. |

**Coefficient grid**
| Param | Default | Notes |
|---|---|---|
| `coef_list` | `[-10, -5, 5, 10]` | Explicit list. Baseline `0` is added automatically. |
| `coef_range` | `null` | `[start, end, step]` or `[[s,e,st], ...]` for multi-segment. Overrides `coef_list`. |

**Vector normalization**
| Param | Default | Notes |
|---|---|---|
| `normalize_vector` | `false` | Normalize the DIM vector before applying coefs. |
| `norm_type` | `unit` | `unit` (L2=1) or `std` (rescale to activation std). |

When normalized, coef=1 ≈ one unit of vector magnitude — interpretable across
layers.

**Few-shots & framing**
| Param | Default | Notes |
|---|---|---|
| `num_shots` | 5 | Few-shot examples per question. `0` = zero-shot. |
| `fewshot_source` | `""` | Path to `data/fewshots/*.yaml`. Ignored in dataset-capture mode. |
| `shuffle_choices` | `true` | Randomize MCF label assignment per question. |

**Performance & resumption**
| Param | Default | Notes |
|---|---|---|
| `max_length` | 2048 | Tokenizer max sequence length. |
| `coef_batch_size` | 0 | `0` = all coefs in one forward. Set `1`/`2` if OOM. |
| `resume` | `true` | Skip layers whose results CSV already exists. |
| `verbose_every` | 20 | Print progress every N questions. |

**Qualitative examples**
| Param | Default | Notes |
|---|---|---|
| `generate_examples` | `true` | Free-text completions per (layer, coef). Slow. |
| `n_examples` | 5 | Sample size per (layer, coef). |
| `max_new_tokens` | 80 | Generation length cap. |

#### `output`
| Param | Default | Notes |
|---|---|---|
| `base_dir` | `results/` | Results go to `<base_dir>/<experiment_id>/`. |

---

## CLI Overrides

These take precedence over YAML without modifying it:

```bash
python experiments/runner.py configs/x.yaml \
    --layers 0-5 7 10           # override sweep.layers
    --base-dir /tmp/results     # override output.base_dir
    --examples / --no-examples  # toggle qualitative generation
```

`--layers` accepts ints and ranges: `--layers 0 1 2` or `--layers 0-3 7 10-11`.

---

## What Gets Logged Where

| Thing | File on disk | W&B |
|---|---|---|
| Per-question scores | `mcf/layer_N_results.csv`, `cf/layer_N_results.csv` | (not logged) |
| Per-(layer, coef) summary | `mcf/combined_summary.csv`, `cf/combined_summary.csv` | `mcf/coef_<v>/<metric>`, plus `mcf/layer_table` (step=layer) |
| Resolved config | `config.yaml` (snapshot) | `config.*` in the run |
| Human-readable summary | `experiment_details.txt` | (not logged) |
| Free-text generations | `examples/examples.csv` | (not logged — could add via `log_artifact`) |

**Coefficient key encoding** — W&B keys can't contain `+`/`-`, so coef values
get encoded as `p`/`n`. Examples: `coef_p5` = +5, `coef_n10` = −10,
`coef_p0.25` = +0.25, `coef_n0` = 0.

---

## Common Gotchas

- **`wandb` not installed** → logger prints `enabled in config but wandb
  package not installed — skipping` and continues. Experiment still runs.
- **Not logged in** → `wandb.init` will hang. `Ctrl-C`, then `wandb login`, or
  set `mode: offline`.
- **OOM** → set `coef_batch_size: 1` in YAML (default 0 = all coefs in one
  forward). For Llama-3-8B on 24 GB cards with a 23-coef sweep, you'll likely
  want this.
- **Wrong experiment_id in W&B** → check no other process is holding a W&B run
  open. `reinit=True` is set so sequential runs should work.
- **Smoke test polluting the real project** → `smoke_test.yaml` and
  `exp_test_wandb.yaml` have `enabled: false` / `mode: offline` respectively.
  Don't flip these without thinking.
- **CF target-only sweeps** (no false-cols dataset) take a different code path
  — they read `combined_summary.csv` per layer instead of going through the
  callback. Works, but mirror any callback changes here too.

---

## Extending the Logging

A few things `wandb_logger.py` supports but the runner doesn't call yet:

- **`log_artifact(run, path, name, type)`** — upload CSVs / plots / the full
  `out_dir` as a W&B artifact. Add a call before `finish_run` in `runner.py`
  if you want results attached to the run.
- **`log_combined_summary(run, df, prefix)`** — one-shot log of an entire
  combined_summary table. Useful as a fallback if the per-layer callback
  didn't fire.
