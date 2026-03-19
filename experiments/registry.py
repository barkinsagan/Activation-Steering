"""
Model and dataset loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.config import ExperimentConfig


# =============================================================================
# Model loading
# =============================================================================

def load_model(cfg: ExperimentConfig):
    """Load HuggingFace model and tokenizer from config."""
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[cfg.model.dtype]

    print(f"Loading model: {cfg.model.name}  (dtype={cfg.model.dtype}, device={cfg.model.device})")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        device_map=cfg.model.device,
    )
    model.eval()

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


# =============================================================================
# Dataset loading
# =============================================================================

def load_eval_dataset(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the evaluation CSV.

    Returns:
        df:           DataFrame with at least 'prompt' and 'target' columns.
                      May also have 'false1', 'false2', 'false3' columns.
        false_cols:   List of false column names found (e.g. ['false1', 'false2'])
    """
    df = pd.read_csv(cfg.dataset.eval_path)

    required = {"prompt", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eval CSV missing required columns: {missing}")

    false_cols = [c for c in ["false1", "false2", "false3"] if c in df.columns]

    print(f"Eval dataset: {len(df)} rows | false cols: {false_cols or 'none'}")
    return df, false_cols


def load_prompts(path: str) -> List[str]:
    """Load prompts from a plain text file (one per line, blank lines skipped)."""
    lines = Path(path).read_text().splitlines()
    prompts = [l.strip() for l in lines if l.strip()]
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def load_steering_prompts(cfg: ExperimentConfig) -> Tuple[List[str], List[str]]:
    """Load positive and negative steering prompts."""
    pos = load_prompts(cfg.dataset.positive_prompts_path)
    neg = load_prompts(cfg.dataset.negative_prompts_path)
    return pos, neg
