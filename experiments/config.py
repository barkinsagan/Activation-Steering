"""
Experiment configuration: YAML loading and validation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# =============================================================================
# Sub-configs
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    dtype: str = "float16"       # float16 | bfloat16 | float32
    device: str = "cuda"


@dataclass
class DatasetConfig:
    eval_path: str                       # CSV with prompt, target, [false1, false2, ...]
    positive_prompts_path: str           # one prompt per line
    negative_prompts_path: str           # one prompt per line


@dataclass
class SweepConfig:
    formulation: str = "both"            # mcf | cf | both
    layers: Optional[List[int]] = None   # None = all layers
    coef_list: List[float] = field(default_factory=lambda: [-10, -5, 5, 10])
    coef_range: Optional[List[float]] = None  # [start, end, step] — overrides coef_list if set
    token_position: str = "last"         # last | mean
    normalize_vector: bool = False
    norm_type: str = "unit"              # unit | std
    task_prefix: str = "question"        # question | goal | fill_in_the_blank | continuation
    cf_normalization: str = "character"  # none | token | character | pmi
    num_shots: int = 5
    fewshot_source: str = ""             # path to data/fewshots/*.yaml (empty = zero-shot)
    shuffle_choices: bool = True         # randomise MCF label assignment per question
    layer_name_pattern: str = "model.layers.{layer_idx}"
    max_length: int = 2048
    verbose_every: int = 20
    resume: bool = True
    coef_batch_size: int = 0        # 0 = all coefs in one batch; set smaller if OOM
    generate_examples: bool = True   # generate qualitative text samples per layer/coef
    n_examples: int = 5              # number of questions to generate per layer/coef
    max_new_tokens: int = 80         # max tokens to generate per example


@dataclass
class OutputConfig:
    base_dir: str = "results/"


# =============================================================================
# Top-level config
# =============================================================================

@dataclass
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig
    dataset: DatasetConfig
    sweep: SweepConfig
    output: OutputConfig

    @property
    def output_dir(self) -> Path:
        return Path(self.output.base_dir) / self.experiment_id


# =============================================================================
# Loader
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment YAML config file.

    If the YAML contains a ``base:`` key, the referenced file is loaded first
    and the current file's values are deep-merged on top (override wins).
    """
    path = Path(path)
    if not path.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "base" in raw:
        base_path = Path(raw.pop("base"))
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        if not base_path.exists():
            print(f"Base config not found: {base_path}", file=sys.stderr)
            sys.exit(1)
        with open(base_path) as f:
            base_raw = yaml.safe_load(f)
        # Base must not itself have a 'base' key (no chaining for simplicity)
        base_raw.pop("base", None)
        raw = _deep_merge(base_raw, raw)

    try:
        model = ModelConfig(**raw["model"])
        dataset = DatasetConfig(**raw["dataset"])
        sweep_raw = raw.get("sweep", {})
        # Resolve coef_range → coef_list before constructing SweepConfig
        # Supports two formats:
        #   single segment:  coef_range: [start, end, step]
        #   multi-segment:   coef_range: [[start, end, step], [start, end, step], ...]
        if "coef_range" in sweep_raw and sweep_raw["coef_range"] is not None:
            segments = sweep_raw["coef_range"]
            # Normalise single segment to list of segments
            if not isinstance(segments[0], list):
                segments = [segments]
            coefs: list = []
            for start, end, step in segments:
                n = round((end - start) / step) + 1
                for i in range(n):
                    v = round(start + i * step, 10)
                    if not coefs or v != coefs[-1]:   # deduplicate segment boundaries
                        coefs.append(v)
            sweep_raw["coef_list"] = coefs
        sweep = SweepConfig(**sweep_raw)
        output = OutputConfig(**raw.get("output", {}))
        experiment_id = raw["experiment_id"]
    except (KeyError, TypeError) as e:
        print(f"Config error in {path}: {e}", file=sys.stderr)
        sys.exit(1)

    cfg = ExperimentConfig(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
        sweep=sweep,
        output=output,
    )

    _validate(cfg, path)
    return cfg


def _validate(cfg: ExperimentConfig, path: Path):
    """Check required files exist and values are valid."""
    errors = []

    # Dataset files
    for attr, label in [
        ("eval_path", "eval_path"),
        ("positive_prompts_path", "positive_prompts_path"),
        ("negative_prompts_path", "negative_prompts_path"),
    ]:
        p = Path(getattr(cfg.dataset, attr))
        if not p.exists():
            errors.append(f"  {label}: file not found: {p}")

    # Enum checks
    valid_formulation = {"mcf", "cf", "both"}
    if cfg.sweep.formulation not in valid_formulation:
        errors.append(f"  formulation must be one of {valid_formulation}")

    valid_position = {"last", "mean"}
    if cfg.sweep.token_position not in valid_position:
        errors.append(f"  token_position must be one of {valid_position}")

    valid_dtype = {"float16", "bfloat16", "float32"}
    if cfg.model.dtype not in valid_dtype:
        errors.append(f"  dtype must be one of {valid_dtype}")

    valid_prefix = {"question", "goal", "fill_in_the_blank", "continuation"}
    if cfg.sweep.task_prefix not in valid_prefix:
        errors.append(f"  task_prefix must be one of {valid_prefix}")

    valid_cf_norm = {"none", "token", "character", "pmi"}
    if cfg.sweep.cf_normalization not in valid_cf_norm:
        errors.append(f"  cf_normalization must be one of {valid_cf_norm}")

    if cfg.sweep.fewshot_source:
        from pathlib import Path as _Path
        if not _Path(cfg.sweep.fewshot_source).exists():
            errors.append(f"  fewshot_source: file not found: {cfg.sweep.fewshot_source}")

    if errors:
        print(f"\nValidation errors in {path}:")
        for e in errors:
            print(e)
        sys.exit(1)
