"""
Single-token completion scoring for steering vector experiments.

For each (layer, question, coefficient), logs:
  - logprob:       log P(first_token_of_target | prompt)
  - rank:          vocab rank of target's first token (1 = most probable)
  - delta_logprob: logprob(steered) - logprob(base)
  - rank_change:   base_rank - steered_rank  (positive = rank improved)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Text Formatting Utilities  (mirrors token_completion_test.py)
# =============================================================================

def _clean(s: Any) -> str:
    return str(s).strip()


def ensure_trailing_space(s: str) -> str:
    return str(s).rstrip() + " "


def ensure_leading_space(s: str) -> str:
    s = _clean(s)
    return " " + s if s else ""


@dataclass(frozen=True)
class PromptTargetFormatter:
    """
    Canonicalize prompts and answer candidates.

    Styles:
        - "qa":   Question: {q}\\nAnswer: format
        - "mmlu": {q}\\nAnswer: format
        - "colon": {q}: format
    """
    style: str = "qa"

    def format_prompt(self, question: str) -> str:
        q = _clean(question)
        if self.style == "qa":
            return ensure_trailing_space(f"Question: {q}\nAnswer:")
        if self.style == "mmlu":
            return ensure_trailing_space(f"{q}\nAnswer:")
        if self.style == "colon":
            return ensure_trailing_space(f"{q}:")
        raise ValueError(f"Unknown style={self.style!r}")

    def format_target(self, target: str) -> str:
        """Add leading space for proper LLaMA/SentencePiece tokenization."""
        return ensure_leading_space(target)


# =============================================================================
# Single Token Scorer
# =============================================================================

class SingleTokenScorer:
    """
    For a given (prompt, continuation) pair, computes:
      - The ID and string of the continuation's first token
      - log P(first_token | prompt) via a forward pass on the prompt alone
      - Vocab rank of first_token in the next-token distribution (1 = most probable)

    Steering-safe: accepts a ModelWithHooks wrapper or any HF model as forward_fn.
    """

    def __init__(
        self,
        forward_fn: Any,
        tokenizer: Any,
        max_length: int = 1024,
        device: Optional[torch.device] = None,
    ):
        self.forward_fn = forward_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = self._infer_device(forward_fn, device)
        self._try_set_eval(forward_fn)

    @staticmethod
    def _infer_device(forward_fn: Any, device: Optional[torch.device]) -> torch.device:
        if device is not None:
            return device
        if hasattr(forward_fn, "device"):
            return getattr(forward_fn, "device")
        if hasattr(forward_fn, "model") and hasattr(forward_fn.model, "device"):
            return getattr(forward_fn.model, "device")
        return torch.device("cpu")

    @staticmethod
    def _try_set_eval(forward_fn: Any) -> None:
        if hasattr(forward_fn, "eval"):
            try:
                forward_fn.eval()
            except Exception:
                pass

    @torch.no_grad()
    def score(self, prompt: str, continuation: str) -> Dict[str, Any]:
        """
        Compute first-token log prob and vocab rank.

        Strategy:
          1. Tokenize (prompt + continuation) to identify first continuation token.
          2. Forward pass on prompt alone to get next-token distribution.
          3. Read log P and rank for that token.

        Returns dict with:
            first_token_id  (int or None if empty continuation)
            first_token_str (str)
            logprob         (float)
            rank            (int, 1-based; -1 if empty continuation)
        """
        # --- Tokenize prompt (to learn its token length) ---
        prompt_tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_ids = prompt_tok["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        # --- Tokenize prompt+continuation (teacher-forcing style, same as original) ---
        full_tok = self.tokenizer(
            prompt + continuation,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        full_ids = full_tok["input_ids"].to(self.device)
        cont_len = full_ids.shape[1] - prompt_len

        if cont_len <= 0:
            return {
                "first_token_id": None,
                "first_token_str": "",
                "logprob": float("-inf"),
                "rank": -1,
            }

        first_token_id = int(full_ids[0, prompt_len].item())

        # --- Forward pass on prompt only ---
        prompt_mask = prompt_tok.get("attention_mask")
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(self.device)
            outputs = self.forward_fn(input_ids=prompt_ids, attention_mask=prompt_mask)
        else:
            outputs = self.forward_fn(input_ids=prompt_ids)

        # logits: [1, prompt_len, vocab_size]
        last_logits = outputs.logits[0, -1, :]          # [vocab_size]
        log_probs   = torch.log_softmax(last_logits, dim=-1)  # [vocab_size]

        # Log prob of the first continuation token
        target_lp = log_probs[first_token_id]           # scalar tensor (on device)

        # Rank: how many vocab tokens have a strictly higher log prob?
        rank = int((log_probs > target_lp).sum().item()) + 1

        return {
            "first_token_id":  first_token_id,
            "first_token_str": self.tokenizer.decode([first_token_id]),
            "logprob":         float(target_lp.item()),
            "rank":            rank,
        }


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class SingleTokenRecord:
    """One record per (layer, question_id, coef)."""
    layer:           int
    question_id:     int
    coef:            float
    prompt:          str
    target_text:     str
    first_token_id:  int
    first_token_str: str
    logprob:         float
    rank:            int
    # Deltas vs baseline (coef == 0); zero-filled for the baseline row itself
    delta_logprob:   float = 0.0   # logprob(steered) - logprob(base)
    rank_change:     int   = 0     # base_rank - steered_rank (positive = improved)


# =============================================================================
# Logger
# =============================================================================

class SingleTokenLogger:
    """
    Collects SingleTokenRecord entries and serialises them to CSV.

    Saves:
        results.csv  — one row per (layer, question, coef)
        summary.csv  — aggregated per (layer, coef)
    """

    def __init__(self, output_dir: str, auto_timestamp: bool = True):
        if auto_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir) / ts
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.records:   List[SingleTokenRecord] = []
        # (layer, question_id) -> (base_logprob, base_rank)
        self._baselines: Dict[tuple, tuple] = {}

    def log(
        self,
        layer:           int,
        question_id:     int,
        coef:            float,
        prompt:          str,
        target_text:     str,
        first_token_id:  int,
        first_token_str: str,
        logprob:         float,
        rank:            int,
    ) -> SingleTokenRecord:
        key = (layer, question_id)

        if coef == 0.0:
            self._baselines[key] = (logprob, rank)

        base_lp, base_rank = self._baselines.get(key, (logprob, rank))
        delta_logprob = (logprob - base_lp) if coef != 0.0 else 0.0
        rank_change   = (base_rank - rank)   if coef != 0.0 else 0

        rec = SingleTokenRecord(
            layer=layer,
            question_id=question_id,
            coef=coef,
            prompt=prompt,
            target_text=target_text,
            first_token_id=first_token_id,
            first_token_str=first_token_str,
            logprob=logprob,
            rank=rank,
            delta_logprob=delta_logprob,
            rank_change=rank_change,
        )
        self.records.append(rec)
        return rec

    def save_results(self, filename: str = "results.csv") -> Path:
        df = pd.DataFrame([asdict(r) for r in self.records])
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")
        return path

    def save_summary(self, filename: str = "summary.csv") -> Path:
        df = pd.DataFrame([asdict(r) for r in self.records])
        summary = _compute_summary(df)
        path = self.output_dir / filename
        summary.to_csv(path, index=False)
        print(f"Saved summary to {path}")
        return path

    def save_all(self) -> Dict[str, Path]:
        return {
            "results": self.save_results(),
            "summary": self.save_summary(),
        }


# =============================================================================
# Summary Helper
# =============================================================================

def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a results DataFrame by (layer, coef)."""
    rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row: Dict[str, Any] = {
            "layer":        int(layer),
            "coef":         float(coef),
            "n":            len(grp),
            "mean_logprob": grp["logprob"].mean(),
            "std_logprob":  grp["logprob"].std(),
            "mean_rank":    grp["rank"].mean(),
            "median_rank":  grp["rank"].median(),
        }
        if coef != 0.0:
            row.update({
                "mean_delta_logprob":   grp["delta_logprob"].mean(),
                "std_delta_logprob":    grp["delta_logprob"].std(),
                "pct_improved_logprob": (grp["delta_logprob"] > 0).mean(),
                "pct_hurt_logprob":     (grp["delta_logprob"] < 0).mean(),
                "mean_rank_change":     grp["rank_change"].mean(),
                "pct_improved_rank":    (grp["rank_change"] > 0).mean(),
                "pct_hurt_rank":        (grp["rank_change"] < 0).mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "coef"]).reset_index(drop=True)


# =============================================================================
# Per-Layer Experiment Runner
# =============================================================================

def run_single_token_experiment(
    scorer:          SingleTokenScorer,
    dataset,                              # DataFrame with 'prompt' and 'target' columns
    dim_steerer,
    steering_vector: torch.Tensor,
    layer_idx:       int,
    coef_list:       Sequence[float],
    logger:          SingleTokenLogger,
    formatter:       Optional[PromptTargetFormatter] = None,
    verbose_every:   int = 20,
) -> None:
    """
    Score baseline + each coefficient for a single layer and append to logger.

    Baseline (coef=0) is always run first so deltas are available immediately.
    """
    if formatter is None:
        formatter = PromptTargetFormatter(style="qa")

    coefs = [0.0] + [c for c in coef_list if c != 0.0]

    for coef in coefs:
        print(f"\n  [Layer {layer_idx}] coef={coef}")

        if coef != 0.0:
            dim_steerer.apply_steering(steering_vector, coefficient=coef)

        try:
            for i, row in dataset.iterrows():
                prompt = formatter.format_prompt(row["prompt"])
                target = formatter.format_target(row["target"])

                result = scorer.score(prompt, target)

                if result["first_token_id"] is None:
                    print(f"    Warning: empty continuation for question {i}, skipping")
                    continue

                logger.log(
                    layer=layer_idx,
                    question_id=i,
                    coef=coef,
                    prompt=prompt,
                    target_text=target,
                    first_token_id=result["first_token_id"],
                    first_token_str=result["first_token_str"],
                    logprob=result["logprob"],
                    rank=result["rank"],
                )

                if verbose_every and (i + 1) % verbose_every == 0:
                    print(f"    Processed {i + 1}/{len(dataset)}")
        finally:
            if coef != 0.0:
                dim_steerer.reset_steering()


# =============================================================================
# Layer Sweep Entry Point
# =============================================================================

def sweep_layers_single_token(
    model,
    tokenizer,
    dataset,                              # DataFrame with 'prompt' and 'target' columns
    positive_prompts:   List[str],
    negative_prompts:   List[str],
    coef_list:          Sequence[float],
    out_dir:            str = "./single_token_sweep",
    layers:             Optional[List[int]] = None,
    token_position:     str = "last",
    normalize_vector:   bool = False,
    norm_type:          str = "unit",
    formatter_style:    str = "qa",
    layer_name_pattern: str = "model.layers.{layer_idx}",
    verbose_every:      int = 20,
    resume:             bool = True,
    start_layer:        Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sweep steering vectors across layers, logging single-token metrics.

    For each (layer, question, coef) the output CSV records:
        layer, question_id, coef,
        prompt, target_text, first_token_id, first_token_str,
        logprob, rank,
        delta_logprob, rank_change

    Args:
        model:               HuggingFace model
        tokenizer:           HuggingFace tokenizer
        dataset:             DataFrame with 'prompt' and 'target' columns
        positive_prompts:    Prompts representing the positive concept
        negative_prompts:    Prompts representing the negative concept
        coef_list:           Steering coefficients (0.0 for baseline added automatically)
        out_dir:             Output directory
        layers:              Layer indices to sweep (default: all hidden layers)
        token_position:      Activation extraction position ("last" or "mean")
        normalize_vector:    Whether to L2/std-normalise the steering vector
        norm_type:           "unit" or "std"
        formatter_style:     Prompt style ("qa", "mmlu", "colon")
        layer_name_pattern:  Pattern for layer names, use {layer_idx} placeholder
        verbose_every:       Print progress every N questions
        resume:              Skip layers whose per-layer CSV already exists
        start_layer:         If set, delete and reprocess all layers >= this index
                             (use this to recover from a partial/corrupted write)

    Returns:
        Dict with keys: logger, combined_results, combined_summary, output_dir
    """
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_with_hooks = ModelWithHooks(model)
    scorer    = SingleTokenScorer(forward_fn=model_with_hooks, tokenizer=tokenizer)
    formatter = PromptTargetFormatter(style=formatter_style)
    logger    = SingleTokenLogger(output_dir=str(out_path), auto_timestamp=False)

    if layers is None:
        num_layers = model.config.num_hidden_layers
        layers = list(range(num_layers))

    layer_dfs: List[pd.DataFrame] = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print("=" * 60)

        layer_result_path = out_path / f"layer_{layer_idx}_results.csv"

        # --- start_layer: delete and reprocess from this layer onward ---
        if start_layer is not None and layer_idx >= start_layer:
            if layer_result_path.exists():
                layer_result_path.unlink()
                print(f"  [start_layer={start_layer}] Deleted existing {layer_result_path.name}, will reprocess.")

        # --- Resume: load previously completed layer ---
        if resume and layer_result_path.exists():
            df = pd.read_csv(layer_result_path)
            print(f"Layer {layer_idx} already completed ({len(df)} records). Loading from disk, skipping.")
            layer_dfs.append(df)
            # Re-populate baseline cache so deltas remain consistent
            for _, r in df[df["coef"] == 0.0].iterrows():
                key = (int(r["layer"]), int(r["question_id"]))
                logger._baselines[key] = (float(r["logprob"]), int(r["rank"]))
            continue

        # --- Build steering vector for this layer ---
        layer_name = layer_name_pattern.format(layer_idx=layer_idx)

        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=token_position,
        )

        print(f"\nCapturing activations for layer {layer_idx}...")
        dim_steerer.capture_positive_activations(positive_prompts)
        dim_steerer.capture_negative_activations(negative_prompts)
        steering_vector = dim_steerer.compute_steering_vector(
            normalize=normalize_vector,
            norm_type=norm_type,
        )

        # --- Run experiment, collect into shared logger ---
        start_idx = len(logger.records)

        run_single_token_experiment(
            scorer=scorer,
            dataset=dataset,
            dim_steerer=dim_steerer,
            steering_vector=steering_vector,
            layer_idx=layer_idx,
            coef_list=coef_list,
            logger=logger,
            formatter=formatter,
            verbose_every=verbose_every,
        )

        # --- Persist per-layer results immediately ---
        layer_records = logger.records[start_idx:]
        layer_df = pd.DataFrame([asdict(r) for r in layer_records])
        layer_df.to_csv(layer_result_path, index=False)
        print(f"\n>>> Layer {layer_idx} COMPLETE — {len(layer_df)} records written to {layer_result_path}")

        layer_dfs.append(layer_df)

        # Checkpoint combined files after every layer so a crash loses at most one layer
        combined_checkpoint = pd.concat(layer_dfs, ignore_index=True)
        combined_checkpoint.to_csv(out_path / "combined_results.csv", index=False)
        _compute_summary(combined_checkpoint).to_csv(out_path / "combined_summary.csv", index=False)
        print(f"    Checkpointed combined_results.csv and combined_summary.csv ({len(combined_checkpoint)} total records, {len(layer_dfs)} layers done)")

        dim_steerer.cleanup()

    # --- Final combined outputs ---
    combined_results = None
    combined_summary = None

    if layer_dfs:
        combined_results = pd.concat(layer_dfs, ignore_index=True)
        combined_results.to_csv(out_path / "combined_results.csv", index=False)
        print(f"\nSaved combined results to {out_path / 'combined_results.csv'}")

        combined_summary = _compute_summary(combined_results)
        combined_summary.to_csv(out_path / "combined_summary.csv", index=False)
        print(f"Saved combined summary to {out_path / 'combined_summary.csv'}")

    return {
        "logger":           logger,
        "combined_results": combined_results,
        "combined_summary": combined_summary,
        "output_dir":       out_path,
    }
