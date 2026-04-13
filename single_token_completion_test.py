"""
Single-token completion scoring for steering vector experiments — MCF formulation.

Implements OLMES-compliant MCF evaluation:
  - Full labeled prompt with A/B/C/D choices injected
  - Scores log P(" A" | prompt), log P(" B" | prompt), ..., picks winner
  - OLMESFormatter handles prompt construction, choice shuffling, few-shot prepending

Reference: https://github.com/allenai/olmes
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from olmes.formatter import OLMESFormatter, MCFFormattedRow, build_formatter

LABELS = ["A", "B", "C", "D"]


# =============================================================================
# Single Token Scorer
# =============================================================================

class MCFScorer:
    """
    OLMES MCF scorer: scores log P(" A"|prompt), log P(" B"|prompt), ...
    and returns the label with the highest log-prob.

    Steering-safe: accepts a ModelWithHooks wrapper or any HF model as forward_fn.
    """

    def __init__(
        self,
        forward_fn: Any,
        tokenizer: Any,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ):
        self.forward_fn = forward_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = self._infer_device(forward_fn, device)
        self._try_set_eval(forward_fn)
        self._label_token_ids = self._resolve_label_tokens()

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

    def _resolve_label_tokens(self) -> Dict[str, int]:
        """
        Map each label letter to a single vocab token ID.

        OLMES uses " A", " B", " C", " D" (space-prefixed) so the token is
        identical regardless of position in the sequence.
        """
        label_ids: Dict[str, int] = {}
        for label in LABELS:
            # Encode with leading space
            ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(ids) == 1:
                label_ids[label] = ids[0]
            else:
                # Fallback: try without space, then take first token
                ids_no_space = self.tokenizer.encode(label, add_special_tokens=False)
                label_ids[label] = ids_no_space[0]
                print(
                    f"Warning: label ' {label}' tokenises to {len(ids)} tokens; "
                    f"using first token id={ids_no_space[0]}"
                )
        return label_ids

    @torch.no_grad()
    def score_mcf(self, prompt: str, correct_label: str) -> Dict[str, Any]:
        """
        Forward pass on prompt, score each label token, return winner.

        Returns:
            label_logprobs:       {"A": float, "B": float, "C": float, "D": float}
            predicted_label:      label with highest log-prob
            correct_label:        ground truth label
            correct:              predicted_label == correct_label
            correct_label_logprob: log-prob of the correct label
            correct_label_rank:   rank of correct label among A/B/C/D (1 = best)
        """
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            outputs = self.forward_fn(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.forward_fn(input_ids=input_ids)

        last_logits = outputs.logits[0, -1, :]              # [vocab_size]
        log_probs   = torch.log_softmax(last_logits, dim=-1)

        label_logprobs: Dict[str, float] = {
            label: log_probs[token_id].item()
            for label, token_id in self._label_token_ids.items()
        }

        predicted_label = max(label_logprobs, key=label_logprobs.__getitem__)
        sorted_lps = sorted(label_logprobs.values(), reverse=True)
        correct_lp = label_logprobs[correct_label]
        correct_rank = sorted_lps.index(correct_lp) + 1

        return {
            "label_logprobs":        label_logprobs,
            "predicted_label":       predicted_label,
            "correct_label":         correct_label,
            "correct":               predicted_label == correct_label,
            "correct_label_logprob": correct_lp,
            "correct_label_rank":    correct_rank,
        }


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class SingleTokenRecord:
    """One record per (layer, question_id, coef) — OLMES MCF formulation."""
    layer:                 int
    question_id:           int
    coef:                  float
    prompt:                str
    correct_label:         str    # ground truth: "A", "B", "C", or "D"
    predicted_label:       str    # model prediction
    correct:               bool   # predicted_label == correct_label
    logprob_A:             float
    logprob_B:             float
    logprob_C:             float
    logprob_D:             float
    correct_label_logprob: float  # log-prob of the ground-truth label
    correct_label_rank:    int    # rank of correct label among A–D (1 = best)
    # Deltas vs baseline (coef == 0); zero-filled for baseline rows
    delta_correct_logprob: float = 0.0   # correct_label_logprob(steered) - baseline
    rank_change:           int   = 0     # base_rank - steered_rank (positive = improved)


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
        layer:                 int,
        question_id:           int,
        coef:                  float,
        prompt:                str,
        correct_label:         str,
        predicted_label:       str,
        correct:               bool,
        label_logprobs:        Dict[str, float],
        correct_label_logprob: float,
        correct_label_rank:    int,
    ) -> SingleTokenRecord:
        key = (layer, question_id)

        if coef == 0.0:
            self._baselines[key] = (correct_label_logprob, correct_label_rank)

        base_lp, base_rank = self._baselines.get(key, (correct_label_logprob, correct_label_rank))
        delta_correct_logprob = (correct_label_logprob - base_lp) if coef != 0.0 else 0.0
        rank_change           = (base_rank - correct_label_rank)   if coef != 0.0 else 0

        rec = SingleTokenRecord(
            layer=layer,
            question_id=question_id,
            coef=coef,
            prompt=prompt,
            correct_label=correct_label,
            predicted_label=predicted_label,
            correct=correct,
            logprob_A=label_logprobs.get("A", float("-inf")),
            logprob_B=label_logprobs.get("B", float("-inf")),
            logprob_C=label_logprobs.get("C", float("-inf")),
            logprob_D=label_logprobs.get("D", float("-inf")),
            correct_label_logprob=correct_label_logprob,
            correct_label_rank=correct_label_rank,
            delta_correct_logprob=delta_correct_logprob,
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
            "layer":                       int(layer),
            "coef":                        float(coef),
            "n":                           len(grp),
            "accuracy":                    grp["correct"].mean(),
            "mean_correct_label_logprob":  grp["correct_label_logprob"].mean(),
            "std_correct_label_logprob":   grp["correct_label_logprob"].std(),
            "mean_correct_label_rank":     grp["correct_label_rank"].mean(),
        }
        if coef != 0.0:
            row.update({
                "mean_delta_correct_logprob":  grp["delta_correct_logprob"].mean(),
                "std_delta_correct_logprob":   grp["delta_correct_logprob"].std(),
                "pct_improved_logprob":        (grp["delta_correct_logprob"] > 0).mean(),
                "pct_hurt_logprob":            (grp["delta_correct_logprob"] < 0).mean(),
                "mean_rank_change":            grp["rank_change"].mean(),
                "pct_improved_rank":           (grp["rank_change"] > 0).mean(),
                "pct_hurt_rank":               (grp["rank_change"] < 0).mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "coef"]).reset_index(drop=True)


# =============================================================================
# Per-Layer Experiment Runner
# =============================================================================

def run_mcf_experiment(
    scorer:          MCFScorer,
    dataset,
    dim_steerer,
    steering_vector: torch.Tensor,
    layer_idx:       int,
    coef_list:       Sequence[float],
    logger:          SingleTokenLogger,
    formatter:       OLMESFormatter,
    verbose_every:   int = 20,
) -> None:
    """
    Score baseline + each coefficient for a single layer (MCF formulation).
    Baseline (coef=0) is always run first so deltas are available immediately.
    """
    coefs = [0.0] + [c for c in coef_list if c != 0.0]

    for coef in coefs:
        print(f"\n  [MCF Layer {layer_idx}] coef={coef}")

        if coef != 0.0:
            dim_steerer.apply_steering(steering_vector, coefficient=coef)

        try:
            for i, row in dataset.iterrows():
                mcf_row: MCFFormattedRow = formatter.format_mcf(row, question_idx=i)
                result = scorer.score_mcf(mcf_row.prompt, mcf_row.correct_label)

                logger.log(
                    layer=layer_idx,
                    question_id=i,
                    coef=coef,
                    prompt=mcf_row.prompt,
                    correct_label=result["correct_label"],
                    predicted_label=result["predicted_label"],
                    correct=result["correct"],
                    label_logprobs=result["label_logprobs"],
                    correct_label_logprob=result["correct_label_logprob"],
                    correct_label_rank=result["correct_label_rank"],
                )

                if verbose_every and (i + 1) % verbose_every == 0:
                    print(f"    Processed {i + 1}/{len(dataset)}")
        finally:
            if coef != 0.0:
                dim_steerer.reset_steering()


# =============================================================================
# Layer Sweep Entry Point
# =============================================================================

def sweep_layers_mcf(
    model,
    tokenizer,
    dataset,
    positive_prompts:   List[str],
    negative_prompts:   List[str],
    coef_list:          Sequence[float],
    formatter:          OLMESFormatter,
    out_dir:            str = "./mcf_sweep",
    layers:             Optional[List[int]] = None,
    token_position:     str = "last",
    normalize_vector:   bool = False,
    norm_type:          str = "unit",
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
        formatter:           OLMESFormatter instance for MCF prompt construction
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
    scorer = MCFScorer(forward_fn=model_with_hooks, tokenizer=tokenizer)
    logger = SingleTokenLogger(output_dir=str(out_path), auto_timestamp=False)

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
                logger._baselines[key] = (float(r["correct_label_logprob"]), int(r["correct_label_rank"]))
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

        run_mcf_experiment(
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
