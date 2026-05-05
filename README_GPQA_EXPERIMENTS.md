# Running GPQA Steering Vector Experiments

This guide walks you through running steering vector experiments on the GPQA (Graduate-Level Google-Proof Q&A) benchmark from scratch. No prior knowledge of the codebase is required.

> **Running a different dataset?** See [README.md](README.md) for the project overview and links to other experiment guides.

---

## What This Experiment Does

We compute a **difference-in-means (DIM) steering vector** — a direction in the model's activation space that points from "general knowledge" toward "expert scientific reasoning." We then apply this vector at different layers and strengths during inference and measure how it affects the model's accuracy on hard science questions.

**Three phases:**
1. **Steering** — capture model activations on GPQA questions (positive) and MMLU questions (negative), compute the DIM vector
2. **Evaluation** — score the model on the remaining GPQA questions at each layer × coefficient, using 5-shot MMLU prompts
3. **Output** — per-layer accuracy tables and summary CSVs saved to your results directory

---

## Requirements

- Python 3.9+
- A CUDA GPU (16GB+ VRAM recommended for Llama-3-8B in float16)
- A HuggingFace account with access to `meta-llama/Meta-Llama-3-8B`
- The `datasets` and `transformers` packages installed

**Install dependencies:**
```bash
pip install torch transformers datasets pandas numpy pyyaml
```

**HuggingFace login** (required to download Llama-3-8B):
```bash
huggingface-cli login
```

---

## Step 1 — Clone the Repository

```bash
git clone <repo_url>
cd steering_project
```

---

## Step 2 — Prepare the Data

You need to run two data preparation scripts. These download the datasets from HuggingFace and save them as CSVs in `data/eval/`.

### 2a. Generate GPQA CSVs

```bash
python data/prepare/convert_gpqa.py --all --general --sweep_n 400 --steering_n 50
```

This creates one CSV per domain plus a combined general-science CSV:

| File | Contents |
|------|----------|
| `data/eval/gpqa_main_biology_sweep.csv` | ~70 Biology questions |
| `data/eval/gpqa_main_chemistry_sweep.csv` | ~98 Chemistry questions |
| `data/eval/gpqa_main_physics_sweep.csv` | ~135 Physics questions |
| `data/eval/gpqa_main_general_sweep.csv` | 150 steering rows (50/domain) + ~153 eval rows |

### 2b. Generate MMLU CSVs (negative capture source + few-shot examples)

```bash
python data/prepare/convert_mmlu.py --subjects all --merge --out data/eval/mmlu_nonmed_sweep.csv
```

This creates `data/eval/mmlu_nonmed_sweep.csv`. It serves two purposes automatically:
- **Rows 0–4** → 5-shot examples prepended to every eval question
- **Rows 5–104** → negative captures used to compute the DIM steering vector

> **Note:** You only need to run these preparation steps once. After that, re-running experiments just re-uses the existing CSVs.

---

## Step 3 — Run an Experiment

```bash
python experiments/runner.py configs/<config_name>.yaml
```

### Available configs

| Config file | Description | Steering rows | Eval rows |
|---|---|---|---|
| `exp_20260505_gpqa_biology_llama8b.yaml` | Biology only | 50 | ~20 |
| `exp_20260505_gpqa_chemistry_llama8b.yaml` | Chemistry only | 50 | ~48 |
| `exp_20260505_gpqa_physics_llama8b.yaml` | Physics only | 50 | ~85 |
| `exp_20260505_gpqa_general_llama8b.yaml` | All 3 domains combined | 150 (50 each) | ~153 |

**Example — run the Physics experiment:**
```bash
python experiments/runner.py configs/exp_20260505_gpqa_physics_llama8b.yaml
```

**Run all four sequentially:**
```bash
python experiments/runner.py configs/exp_20260505_gpqa_*.yaml
```

**Override the output directory** (e.g. for Google Drive on Colab):
```bash
python experiments/runner.py configs/exp_20260505_gpqa_physics_llama8b.yaml \
  --base-dir /content/drive/MyDrive/my_results/
```

**Override layers without editing the config:**
```bash
python experiments/runner.py configs/exp_20260505_gpqa_physics_llama8b.yaml \
  --layers 15 20 25
```

---

## Step 4 — Find Your Results

Results are saved to `results/<experiment_id>/` by default (or your `--base-dir`).

```
results/exp_20260505_gpqa_physics_llama8b/
├── config.yaml                  # snapshot of the config used
├── experiment_details.txt       # human-readable run summary
├── mcf/                         # Multiple-choice formulation results
│   ├── layer_0_results.csv
│   ├── layer_5_results.csv
│   ├── ...
│   ├── combined_results.csv     # all layers merged
│   └── combined_summary.csv    # accuracy by (layer, coef) ← start here
└── cf/                          # Continuation formulation results
    ├── layer_0_results.csv
    ├── ...
    └── combined_summary.csv
```

The most useful file is **`combined_summary.csv`** — it shows accuracy (or log-probability) for every layer × coefficient combination.

---

## Understanding the Config

Every experiment is controlled by a single YAML file. Here are the fields you are most likely to want to change:

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B   # swap for any HuggingFace model
  dtype: float16                      # float16 | bfloat16 | float32
  device: cuda                        # cuda | cpu

dataset:
  eval_path: data/eval/gpqa_main_physics_sweep.csv
  neg_capture_path: data/eval/mmlu_nonmed_sweep.csv
  capture_n: 50          # how many rows from eval_path to use for steering

sweep:
  formulation: both      # mcf | cf | both
  layers: [0, 5, 15, 20, 25, 30]   # which transformer layers to test

  # Coefficients to apply (-2 to 2, 0.25 steps, plus ±2)
  coef_range:
    - [-2,  -1, 1]
    - [-1,   1, 0.25]
    - [ 1,   2, 1]

  num_shots: 5           # few-shot examples during eval (0 = zero-shot)
  normalize_vector: true # normalize the steering vector to unit length
  resume: true           # skip already-completed layers if interrupted

output:
  base_dir: results/     # where to save outputs
```

### Coefficient guide

| Coefficient | Effect |
|---|---|
| `0` | Baseline — no steering applied |
| `> 0` | Steer toward expert science (GPQA direction) |
| `< 0` | Steer away from expert science |
| `±2` | Strong intervention (may degrade fluency) |

---

## Running on Google Colab

1. Mount your Drive and clone the repo (or `git pull` if already cloned)
2. Run the data preparation steps (Step 2 above)
3. Run experiments with `--base-dir` pointing to your Drive so results are saved persistently:

```python
!python experiments/runner.py configs/exp_20260505_gpqa_physics_llama8b.yaml \
  --base-dir /content/drive/MyDrive/SteeringProject/results/
```

If the session disconnects, re-run the same command — `resume: true` in the config means completed layers are skipped automatically.

---

## Creating a New Config

To run on a different model, dataset, or set of layers, copy an existing config and modify it:

```bash
cp configs/exp_20260505_gpqa_physics_llama8b.yaml configs/my_new_experiment.yaml
```

Edit `my_new_experiment.yaml`:
- Set `experiment_id` to a unique name (convention: `exp_YYYYMMDD_shortname`)
- Change `model.name` to a different HuggingFace model if needed
- Change `dataset.eval_path` to point to a different CSV
- Adjust `layers`, `coef_range`, or `formulation` as needed

Then run:
```bash
python experiments/runner.py configs/my_new_experiment.yaml
```

Results will be saved to `results/my_new_experiment/`.
