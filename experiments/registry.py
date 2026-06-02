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

    quant_suffix = " 8-bit" if cfg.model.load_in_8bit else ""
    print(f"Loading model: {cfg.model.name}  (dtype={cfg.model.dtype}{quant_suffix}, device={cfg.model.device})")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": cfg.model.device}
    if cfg.model.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **load_kwargs)
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
    """Load positive and negative steering prompts (text-file mode)."""
    pos = load_prompts(cfg.dataset.positive_prompts_path)
    neg = load_prompts(cfg.dataset.negative_prompts_path)
    return pos, neg


# =============================================================================
# Dataset capture helpers
# =============================================================================

def df_to_fewshot_examples(df: pd.DataFrame, n: int) -> List[dict]:
    """Convert first n rows of an eval DataFrame to OLMESFormatter few-shot format."""
    examples = []
    for _, row in df.head(n).iterrows():
        examples.append({
            "prompt": str(row["prompt"]),
            "choices": [str(row["target"]), str(row["false1"]),
                        str(row["false2"]), str(row["false3"])],
            "correct": 0,
        })
    return examples


def build_capture_prompts_mcf(df: pd.DataFrame, formatter, n: int) -> List[str]:
    """Format first n rows as MCF capture prompts (ending at 'Answer:')."""
    prompts = []
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        mcf_row = formatter.format_mcf(row, question_idx=i)
        prompts.append(mcf_row.prompt)
    return prompts


def build_capture_prompts_cf(df: pd.DataFrame, formatter, n: int) -> List[str]:
    """Format first n rows as CF capture prompts (question + correct answer, no choices)."""
    prompts = []
    for _, row in df.head(n).iterrows():
        cf_row = formatter.format_cf(row)
        prompts.append(cf_row.prompt + cf_row.target)
    return prompts


# =============================================================================
# Random split
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    split_cfg,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int], List[int], List[int]]:
    """Randomly split df into steering, validation, and test sets.

    Returns:
        steering_df, val_df, test_df: the three DataFrames (reset index)
        steer_indices, val_indices, test_indices: original row positions in df
    """
    import numpy as np

    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()

    n_steer = round(n * split_cfg.steering / 100)
    n_val   = round(n * split_cfg.validation / 100)
    # test gets the remainder to absorb rounding drift

    steer_idx = sorted(perm[:n_steer])
    val_idx   = sorted(perm[n_steer: n_steer + n_val])
    test_idx  = sorted(perm[n_steer + n_val:])

    steering_df = df.iloc[steer_idx].reset_index(drop=True)
    val_df      = df.iloc[val_idx].reset_index(drop=True)
    test_df     = df.iloc[test_idx].reset_index(drop=True)

    print(
        f"Random split (seed={seed}): "
        f"{len(steering_df)} steering ({split_cfg.steering}%) / "
        f"{len(val_df)} validation ({split_cfg.validation}%) / "
        f"{len(test_df)} test ({split_cfg.test}%)"
    )
    return steering_df, val_df, test_df, steer_idx, val_idx, test_idx
