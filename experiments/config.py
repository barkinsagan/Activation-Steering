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
    scoring_mode: str = "both"           # single_token | continuation | both
    layers: Optional[List[int]] = None   # None = all layers
    coef_list: List[float] = field(default_factory=lambda: [-10, -5, 5, 10])
    token_position: str = "last"         # last | mean
    normalize_vector: bool = False
    norm_type: str = "unit"              # unit | std
    formatter_style: str = "qa"          # qa | mmlu | colon
    layer_name_pattern: str = "model.layers.{layer_idx}"
    max_length: int = 1024
    verbose_every: int = 20
    resume: bool = True


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

def load_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment YAML config file."""
    path = Path(path)
    if not path.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = yaml.safe_load(f)

    try:
        model = ModelConfig(**raw["model"])
        dataset = DatasetConfig(**raw["dataset"])
        sweep_raw = raw.get("sweep", {})
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
    valid_scoring = {"single_token", "continuation", "both"}
    if cfg.sweep.scoring_mode not in valid_scoring:
        errors.append(f"  scoring_mode must be one of {valid_scoring}")

    valid_position = {"last", "mean"}
    if cfg.sweep.token_position not in valid_position:
        errors.append(f"  token_position must be one of {valid_position}")

    valid_dtype = {"float16", "bfloat16", "float32"}
    if cfg.model.dtype not in valid_dtype:
        errors.append(f"  dtype must be one of {valid_dtype}")

    valid_formatter = {"qa", "mmlu", "colon"}
    if cfg.sweep.formatter_style not in valid_formatter:
        errors.append(f"  formatter_style must be one of {valid_formatter}")

    if errors:
        print(f"\nValidation errors in {path}:")
        for e in errors:
            print(e)
        sys.exit(1)
