"""
Token completion scoring for steering vector experiments — CF (cloze/completion) formulation.

Implements OLMES-compliant CF evaluation:
  - Teacher-forcing log-prob over full answer continuation
  - Four normalisation schemes: none | token | character | pmi
  - OLMESFormatter for prompt construction and few-shot prepending

Reference: https://github.com/allenai/olmes
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from olmes.formatter import OLMESFormatter, CFFormattedRow, build_formatter


# =============================================================================
# Constants
# =============================================================================

CANDIDATE_NAMES = ["target", "false1", "false2", "false3"]
CF_NORM_TO_ACCURACY_COL = {
    "none":      "accuracy_sum",
    "token":     "accuracy_mean",
    "character": "accuracy_char",
    "pmi":       "accuracy_pmi",
}


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class CandidateResult:
    """Result of computing continuation probability for a single candidate."""
    text: str
    token_count: int
    char_count: int                       # len(text) incl. leading space — for character norm
    sum_logprob: float                    # OLMES: none
    mean_logprob: float                   # OLMES: token  (sum / num_tokens)
    char_norm_logprob: float              # OLMES: character  (sum / num_chars)
    first_token_logprob: Optional[float]
    pmi_logprob: Optional[float] = None   # OLMES: pmi  (set externally by sweep)


# =============================================================================
# Continuation Probability Scorer
# =============================================================================

class ContinuationProbability:
    """
    Computes log P(continuation | prompt) via teacher forcing.

    Steering-safe: accepts a ModelWithHooks wrapper or any HF model as forward_fn.

    Returns all OLMES CF normalisation variants in one pass:
      sum_logprob      → OLMES none
      mean_logprob     → OLMES token
      char_norm_logprob→ OLMES character
      pmi_logprob      → OLMES pmi  (computed externally; stored here as Optional)
    """

    def __init__(
        self,
        forward_fn: Callable[..., Any],
        tokenizer: Any,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ):
        self.forward_fn = forward_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = self._infer_device(forward_fn, device)
        self._try_set_eval(forward_fn)

    @staticmethod
    def _infer_device(forward_fn: Any, device: Optional[torch.device]) -> torch.device:
        """Infer device from forward_fn if not explicitly provided."""
        if device is not None:
            return device
        if hasattr(forward_fn, "device"):
            return getattr(forward_fn, "device")
        if hasattr(forward_fn, "model") and hasattr(forward_fn.model, "device"):
            return getattr(forward_fn.model, "device")
        return torch.device("cpu")

    @staticmethod
    def _try_set_eval(forward_fn: Any) -> None:
        """Set eval mode if forward_fn is a torch.nn.Module."""
        if hasattr(forward_fn, "eval"):
            try:
                forward_fn.eval()
            except Exception:
                pass

    @torch.no_grad()
    def continuation_logprob(self, prompt: str, continuation: str) -> CandidateResult:
        """Compute log probability of continuation given prompt."""
        # Tokenize prompt alone (to find prompt_len)
        prompt_tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_ids = prompt_tok["input_ids"].to(self.device)

        # Tokenize prompt+continuation (teacher forcing)
        full_tok = self.tokenizer(
            prompt + continuation,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        full_ids = full_tok["input_ids"].to(self.device)
        full_mask = full_tok.get("attention_mask")
        if full_mask is not None:
            full_mask = full_mask.to(self.device)

        prompt_len = prompt_ids.shape[1]
        cont_len = full_ids.shape[1] - prompt_len

        char_count = len(continuation)  # includes leading space per OLMES spec

        if cont_len <= 0:
            return CandidateResult(
                text=continuation,
                token_count=0,
                char_count=char_count,
                sum_logprob=0.0,
                mean_logprob=0.0,
                char_norm_logprob=0.0,
                first_token_logprob=None,
            )

        # Forward pass
        if full_mask is not None:
            outputs = self.forward_fn(input_ids=full_ids, attention_mask=full_mask)
        else:
            outputs = self.forward_fn(input_ids=full_ids)

        logits = outputs.logits

        # Compute log-probs for each continuation token
        sum_lp = 0.0
        first_lp: Optional[float] = None

        for t in range(cont_len):
            pred_pos = (prompt_len - 1) + t
            if pred_pos < 0 or pred_pos >= logits.shape[1]:
                continue

            token_id = full_ids[0, prompt_len + t]
            log_probs = torch.log_softmax(logits[0, pred_pos], dim=-1)
            lp_t = log_probs[token_id].item()

            if t == 0:
                first_lp = lp_t
            sum_lp += lp_t

        mean_lp = sum_lp / max(cont_len, 1)
        char_norm_lp = sum_lp / max(char_count, 1)

        return CandidateResult(
            text=continuation,
            token_count=int(cont_len),
            char_count=char_count,
            sum_logprob=float(sum_lp),
            mean_logprob=float(mean_lp),
            char_norm_logprob=float(char_norm_lp),
            first_token_logprob=float(first_lp) if first_lp is not None else None,
        )

    def _pick_score(self, r: CandidateResult, mode: ScoreMode) -> float:
        """Extract the appropriate score based on mode."""
        if mode == "sum_logprob":
            return r.sum_logprob
        if mode == "mean_logprob":
            return r.mean_logprob
        if mode == "first_token_logprob":
            return r.first_token_logprob if r.first_token_logprob is not None else -1e30
        raise ValueError(f"Unknown mode: {mode}")

    def relative_probs(
        self,
        prompt: str,
        candidates: List[str],
        mode: ScoreMode = "mean_logprob",
    ) -> Dict[str, Any]:
        """Compute relative probabilities for candidates given a prompt."""
        results = [self.continuation_logprob(prompt, c) for c in candidates]
        scores = [self._pick_score(r, mode) for r in results]

        # Stable softmax
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        Z = sum(exps) if exps else 1.0
        probs = [e / Z for e in exps]

        return {
            "mode": mode,
            "scores": scores,
            "relative_probs": probs,
            "results": results,
        }

    def mcq_relative_probs(
        self,
        prompt: str,
        target: str,
        false1: str,
        false2: str,
        false3: str,
        mode: ScoreMode = "mean_logprob",
    ) -> Dict[str, Any]:
        """Compute relative probabilities for MCQ with 4 candidates."""
        candidates = [target, false1, false2, false3]
        out = self.relative_probs(prompt, candidates, mode=mode)
        out["candidates"] = candidates
        return out


# =============================================================================
# Logging Dataclasses
# =============================================================================

@dataclass
class QuestionResult:
    """Stores all metrics for a single question."""
    question_id: int
    prompt: str
    coef: float

    # Candidate texts
    target_text: str
    false1_text: str
    false2_text: str
    false3_text: str

    # Token counts
    target_token_count: int
    false1_token_count: int
    false2_token_count: int
    false3_token_count: int

    # Sum log-probs (OLMES: none)
    target_sum_lp: float
    false1_sum_lp: float
    false2_sum_lp: float
    false3_sum_lp: float

    # Mean log-probs (OLMES: token)
    target_mean_lp: float
    false1_mean_lp: float
    false2_mean_lp: float
    false3_mean_lp: float

    # Char-normalised log-probs (OLMES: character)
    target_char_norm_lp: float
    false1_char_norm_lp: float
    false2_char_norm_lp: float
    false3_char_norm_lp: float

    # First token log-probs
    target_first_lp: Optional[float]
    false1_first_lp: Optional[float]
    false2_first_lp: Optional[float]
    false3_first_lp: Optional[float]

    # PMI log-probs (OLMES: pmi) — None when not computed
    target_pmi_lp: Optional[float] = None
    false1_pmi_lp: Optional[float] = None
    false2_pmi_lp: Optional[float] = None
    false3_pmi_lp: Optional[float] = None

    # --- Derived (computed in __post_init__) ---
    correct_sum: bool = False
    correct_mean: bool = False
    correct_char: bool = False
    correct_pmi: bool = False

    target_rank_sum: int = 0
    target_rank_mean: int = 0
    target_rank_char: int = 0
    target_rank_pmi: int = 0

    margin_sum: float = 0.0
    margin_mean: float = 0.0
    margin_char: float = 0.0
    margin_pmi: float = 0.0

    max_wrong_sum_lp: float = 0.0
    max_wrong_char_norm_lp: float = 0.0

    def __post_init__(self):
        """Compute derived correctness/rank/margin metrics for all normalisations."""

        def _winner_and_stats(lps: List[float]):
            """Return (correct, rank, margin) from 4 raw log-probs."""
            best_idx = lps.index(max(lps))
            correct = (best_idx == 0)           # index 0 = target
            rank = sorted(lps, reverse=True).index(lps[0]) + 1
            margin = lps[0] - max(lps[1:])
            return correct, rank, margin

        # none (sum)
        sum_lps = [self.target_sum_lp, self.false1_sum_lp,
                   self.false2_sum_lp, self.false3_sum_lp]
        self.correct_sum, self.target_rank_sum, self.margin_sum = _winner_and_stats(sum_lps)
        self.max_wrong_sum_lp = max(sum_lps[1:])

        # token (mean)
        mean_lps = [self.target_mean_lp, self.false1_mean_lp,
                    self.false2_mean_lp, self.false3_mean_lp]
        self.correct_mean, self.target_rank_mean, self.margin_mean = _winner_and_stats(mean_lps)

        # character
        char_lps = [self.target_char_norm_lp, self.false1_char_norm_lp,
                    self.false2_char_norm_lp, self.false3_char_norm_lp]
        self.correct_char, self.target_rank_char, self.margin_char = _winner_and_stats(char_lps)
        self.max_wrong_char_norm_lp = max(char_lps[1:])

        # pmi — only when all four values are present
        pmi_lps = [self.target_pmi_lp, self.false1_pmi_lp,
                   self.false2_pmi_lp, self.false3_pmi_lp]
        if all(v is not None for v in pmi_lps):
            self.correct_pmi, self.target_rank_pmi, self.margin_pmi = _winner_and_stats(
                pmi_lps  # type: ignore[arg-type]
            )

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        Z = sum(exps)
        return [e / Z for e in exps]


@dataclass
class ExperimentMetadata:
    """Stores experiment configuration."""
    experiment_name: str = ""
    model_name: str = ""
    steering_vector_source: str = ""
    layer: Optional[int] = None
    dataset_name: str = ""
    dataset_size: int = 0
    cf_normalization: str = "character"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


# =============================================================================
# Result Logger
# =============================================================================

class SteeringResultLogger:
    """
    Logs steering experiment results to CSV files.

    Saves:
        - detailed_wide.csv: One row per question, all metrics as columns
        - detailed_long.csv: One row per (question, candidate), for plotting
        - summary.csv: Aggregated stats per coefficient
        - metadata.json: Experiment configuration
    """

    def __init__(
        self,
        output_dir: str,
        metadata: Optional[ExperimentMetadata] = None,
        auto_timestamp: bool = True,
    ):
        if auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir) / timestamp
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or ExperimentMetadata()
        self.results: List[QuestionResult] = []
        self.baseline_results: Dict[int, QuestionResult] = {}

    def log_question(
        self,
        question_id: int,
        prompt: str,
        coef: float,
        candidate_results: List[CandidateResult],
    ) -> QuestionResult:
        """Log results for a single question."""
        if len(candidate_results) != 4:
            raise ValueError(f"Expected 4 candidates, got {len(candidate_results)}")

        target, false1, false2, false3 = candidate_results

        result = QuestionResult(
            question_id=question_id,
            prompt=prompt,
            coef=coef,
            target_text=target.text,
            false1_text=false1.text,
            false2_text=false2.text,
            false3_text=false3.text,
            target_token_count=target.token_count,
            false1_token_count=false1.token_count,
            false2_token_count=false2.token_count,
            false3_token_count=false3.token_count,
            target_sum_lp=target.sum_logprob,
            false1_sum_lp=false1.sum_logprob,
            false2_sum_lp=false2.sum_logprob,
            false3_sum_lp=false3.sum_logprob,
            target_mean_lp=target.mean_logprob,
            false1_mean_lp=false1.mean_logprob,
            false2_mean_lp=false2.mean_logprob,
            false3_mean_lp=false3.mean_logprob,
            target_char_norm_lp=target.char_norm_logprob,
            false1_char_norm_lp=false1.char_norm_logprob,
            false2_char_norm_lp=false2.char_norm_logprob,
            false3_char_norm_lp=false3.char_norm_logprob,
            target_first_lp=target.first_token_logprob,
            false1_first_lp=false1.first_token_logprob,
            false2_first_lp=false2.first_token_logprob,
            false3_first_lp=false3.first_token_logprob,
            target_pmi_lp=target.pmi_logprob,
            false1_pmi_lp=false1.pmi_logprob,
            false2_pmi_lp=false2.pmi_logprob,
            false3_pmi_lp=false3.pmi_logprob,
        )

        self.results.append(result)
        if coef == 0.0:
            self.baseline_results[question_id] = result

        return result

    def _compute_deltas(self, result: QuestionResult) -> Dict[str, float]:
        """Compute deltas from baseline for a steered result."""
        baseline = self.baseline_results.get(result.question_id)
        if baseline is None or result.coef == 0.0:
            return {
                "delta_target_sum_lp":       0.0,
                "delta_target_char_norm_lp": 0.0,
                "delta_false1_sum_lp":       0.0,
                "delta_false2_sum_lp":       0.0,
                "delta_false3_sum_lp":       0.0,
                "delta_margin_sum":          0.0,
                "delta_margin_char":         0.0,
                "delta_max_wrong_sum_lp":    0.0,
                "rank_change_sum":           0,
                "rank_change_char":          0,
            }

        return {
            "delta_target_sum_lp":       result.target_sum_lp - baseline.target_sum_lp,
            "delta_target_char_norm_lp": result.target_char_norm_lp - baseline.target_char_norm_lp,
            "delta_false1_sum_lp":       result.false1_sum_lp - baseline.false1_sum_lp,
            "delta_false2_sum_lp":       result.false2_sum_lp - baseline.false2_sum_lp,
            "delta_false3_sum_lp":       result.false3_sum_lp - baseline.false3_sum_lp,
            "delta_margin_sum":          result.margin_sum - baseline.margin_sum,
            "delta_margin_char":         result.margin_char - baseline.margin_char,
            "delta_max_wrong_sum_lp":    result.max_wrong_sum_lp - baseline.max_wrong_sum_lp,
            "rank_change_sum":           baseline.target_rank_sum - result.target_rank_sum,
            "rank_change_char":          baseline.target_rank_char - result.target_rank_char,
        }

    def save_detailed_wide(self, filename: str = "detailed_wide.csv") -> Path:
        """Save detailed results in wide format (one row per question)."""
        rows = []
        for r in self.results:
            row = asdict(r)
            row.update(self._compute_deltas(r))
            rows.append(row)

        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved wide-format results to {path}")
        return path

    def save_detailed_long(self, filename: str = "detailed_long.csv") -> Path:
        """Save detailed results in long format (one row per candidate)."""
        rows = []
        for r in self.results:
            base = self.baseline_results.get(r.question_id)

            for cand_name in CANDIDATE_NAMES:
                sum_lp = getattr(r, f"{cand_name}_sum_lp")
                first_lp = getattr(r, f"{cand_name}_first_lp")
                mean_lp = getattr(r, f"{cand_name}_mean_lp")
                token_count = getattr(r, f"{cand_name}_token_count")
                prob_sum = getattr(r, f"{cand_name}_prob_sum")
                prob_first = getattr(r, f"{cand_name}_prob_first")
                text = getattr(r, f"{cand_name}_text")

                base_sum_lp = getattr(base, f"{cand_name}_sum_lp") if base else sum_lp
                base_first_lp = getattr(base, f"{cand_name}_first_lp") if base else first_lp

                rows.append({
                    "question_id": r.question_id,
                    "coef": r.coef,
                    "candidate": cand_name,
                    "is_target": cand_name == "target",
                    "text": text,
                    "token_count": token_count,
                    "sum_lp": sum_lp,
                    "first_lp": first_lp,
                    "mean_lp": mean_lp,
                    "prob_sum": prob_sum,
                    "prob_first": prob_first,
                    "delta_sum_lp": sum_lp - base_sum_lp if r.coef != 0.0 else 0.0,
                    "delta_first_lp": ((first_lp - base_first_lp)
                                       if (first_lp is not None and base_first_lp is not None and r.coef != 0.0)
                                       else 0.0),
                })

        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved long-format results to {path}")
        return path

    def save_summary(self, filename: str = "summary.csv") -> Path:
        """Save aggregated summary per coefficient."""
        df = pd.DataFrame([asdict(r) for r in self.results])
        delta_rows = [self._compute_deltas(r) for r in self.results]
        df_deltas = pd.DataFrame(delta_rows)
        df = pd.concat([df, df_deltas], axis=1)

        summary_rows = []
        for coef in sorted(df["coef"].unique()):
            subset = df[df["coef"] == coef]

            row = {
                "coef": coef,
                "n_questions": len(subset),
                "accuracy_sum":  subset["correct_sum"].mean(),
                "accuracy_mean": subset["correct_mean"].mean(),
                "accuracy_char": subset["correct_char"].mean(),
                "accuracy_pmi":  subset["correct_pmi"].mean() if "correct_pmi" in subset.columns else None,
                "mean_target_sum_lp":       subset["target_sum_lp"].mean(),
                "mean_target_char_norm_lp": subset["target_char_norm_lp"].mean(),
                "mean_margin_sum":          subset["margin_sum"].mean(),
                "mean_margin_char":         subset["margin_char"].mean(),
                "mean_target_rank_sum":  subset["target_rank_sum"].mean(),
                "mean_target_rank_char": subset["target_rank_char"].mean(),
            }

            if coef != 0.0:
                row.update({
                    "mean_delta_target_sum_lp":       subset["delta_target_sum_lp"].mean(),
                    "std_delta_target_sum_lp":        subset["delta_target_sum_lp"].std(),
                    "mean_delta_target_char_norm_lp": subset["delta_target_char_norm_lp"].mean(),
                    "std_delta_target_char_norm_lp":  subset["delta_target_char_norm_lp"].std(),
                    "pct_improved_sum":  (subset["delta_target_sum_lp"] > 0).mean(),
                    "pct_hurt_sum":      (subset["delta_target_sum_lp"] < 0).mean(),
                    "pct_improved_char": (subset["delta_target_char_norm_lp"] > 0).mean(),
                    "pct_hurt_char":     (subset["delta_target_char_norm_lp"] < 0).mean(),
                    "mean_delta_margin_sum":  subset["delta_margin_sum"].mean(),
                    "mean_delta_margin_char": subset["delta_margin_char"].mean(),
                    "mean_rank_change_sum":   subset["rank_change_sum"].mean(),
                    "mean_rank_change_char":  subset["rank_change_char"].mean(),
                })

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        path = self.output_dir / filename
        summary_df.to_csv(path, index=False)
        print(f"Saved summary to {path}")
        return path

    def save_metadata(self, filename: str = "metadata.json") -> Path:
        """Save experiment metadata."""
        self.metadata.dataset_size = len(set(r.question_id for r in self.results))

        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2)
        print(f"Saved metadata to {path}")
        return path

    def save_all(self) -> Dict[str, Path]:
        """Save all output files."""
        return {
            "detailed_wide": self.save_detailed_wide(),
            "detailed_long": self.save_detailed_long(),
            "summary": self.save_summary(),
            "metadata": self.save_metadata(),
        }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_cf_experiment(
    scorer: ContinuationProbability,
    ml_test,
    dim_steerer,
    steering_vector,
    coef_list: Sequence[float],
    output_dir: str,
    formatter: OLMESFormatter,
    cf_normalization: str = "character",
    metadata: Optional[ExperimentMetadata] = None,
    pmi_baselines: Optional[Dict[int, Dict[str, float]]] = None,
    verbose_every: int = 20,
) -> SteeringResultLogger:
    """
    Run one CF experiment (single layer) with OLMES-compliant scoring.

    PMI baselines (if cf_normalization=="pmi") must be pre-computed by the
    caller and passed in as {question_id: {candidate_name: unconditional_sum_lp}}.
    """
    coefs = [0.0] + [c for c in coef_list if c != 0.0]
    logger = SteeringResultLogger(output_dir, metadata, auto_timestamp=False)
    n = len(ml_test)

    for coef in coefs:
        print(f"\n  [CF] coef={coef}")

        if coef != 0.0:
            dim_steerer.apply_steering(steering_vector, coefficient=coef)

        try:
            for i, row in ml_test.iterrows():
                rr: CFFormattedRow = formatter.format_cf(row)

                cands = [
                    scorer.continuation_logprob(rr.prompt, rr.target),
                    scorer.continuation_logprob(rr.prompt, rr.false1),
                    scorer.continuation_logprob(rr.prompt, rr.false2),
                    scorer.continuation_logprob(rr.prompt, rr.false3),
                ]

                # Attach PMI log-probs when available
                if pmi_baselines and i in pmi_baselines:
                    pb = pmi_baselines[i]
                    for cand, name in zip(cands, CANDIDATE_NAMES):
                        cand.pmi_logprob = cand.sum_logprob - pb[name]

                logger.log_question(
                    question_id=i,
                    prompt=rr.prompt,
                    coef=coef,
                    candidate_results=cands,
                )

                if verbose_every and (i + 1) % verbose_every == 0:
                    print(f"    [coef={coef}] {i + 1}/{n}")
        finally:
            if coef != 0.0:
                dim_steerer.reset_steering()

    logger.save_all()
    return logger


# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_per_question_deltas(
    base_relative_probs: List[List[float]],
    steered_relative_probs: List[List[float]],
) -> np.ndarray:
    """
    Compute per-question improvement from steering.

    Returns:
        Array of shape [num_questions] with delta = steered_target - base_target
    """
    assert len(base_relative_probs) == len(steered_relative_probs)

    deltas = []
    for base_row, steer_row in zip(base_relative_probs, steered_relative_probs):
        deltas.append(steer_row[0] - base_row[0])

    return np.array(deltas)


def compute_average_improvement(
    base_relative_probs: List[List[float]],
    steered_results: Dict[float, List[List[float]]],
) -> tuple:
    """
    Compute average improvement across all steering coefficients.

    Returns:
        Tuple of (avg_improvements dict, all_deltas dict)
    """
    avg_improvements = {}
    all_deltas = {}

    for coef, steered_probs in steered_results.items():
        deltas = compute_per_question_deltas(base_relative_probs, steered_probs)
        all_deltas[coef] = deltas
        avg_improvements[coef] = deltas.mean()

    return avg_improvements, all_deltas


def plot_average_improvement(
    avg_improvements: Dict[float, float],
    title: str = "Average per-question improvement from steering",
    figsize: tuple = (7, 5),
    show: bool = True,
) -> plt.Figure:
    """Plot average improvement vs steering coefficient."""
    coefs = sorted(avg_improvements.keys())
    avg_deltas = [avg_improvements[c] for c in coefs]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(coefs, avg_deltas, marker="o")
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Steering coefficient (α)")
    ax.set_ylabel("Average Δ target probability (steered − base)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


# =============================================================================
# Layer Sweep Runner
# =============================================================================

def _find_summary_csv(layer_dir: Path) -> Optional[Path]:
    """Find summary.csv in a layer directory, checking both direct and timestamp subdirs."""
    if not layer_dir.exists():
        return None
    # Check directly in layer dir
    direct = layer_dir / "summary.csv"
    if direct.exists():
        return direct
    # Check inside timestamp subdirectories
    for subdir in sorted(layer_dir.iterdir(), reverse=True):
        if subdir.is_dir():
            candidate = subdir / "summary.csv"
            if candidate.exists():
                return candidate
    return None


def sweep_layers_cf(
    model,
    tokenizer,
    ml_test_df: pd.DataFrame,
    positive_prompts: List[str],
    negative_prompts: List[str],
    coef_list: Sequence[float],
    formatter: OLMESFormatter,
    cf_normalization: str = "character",
    out_dir: str = "./layer_sweep_results",
    verbose_every: int = 50,
    layers: Optional[List[int]] = None,
    token_position: str = "last",
    normalize_vector: bool = False,
    norm_type: str = "unit",
    layer_name_pattern: str = "model.layers.{layer_idx}",
    resume: bool = True,
    start_layer: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sweep steering vectors across layers and save results.

    Args:
        model: HuggingFace model (will be wrapped with ModelWithHooks)
        tokenizer: HuggingFace tokenizer
        ml_test_df: DataFrame with columns: prompt, target, false1, false2, false3
        positive_prompts: Prompts representing the positive concept
        negative_prompts: Prompts representing the negative concept
        coef_list: Steering coefficients to test (0.0 added automatically for baseline)
        mode: Scoring mode ("first_token_logprob", "mean_logprob", "sum_logprob")
        out_dir: Output directory for results
        verbose_every: Print progress every N rows
        layers: List of layer indices to sweep (defaults to all layers)
        token_position: "last" or "mean" for activation extraction
        normalize_vector: Whether to normalize steering vectors
        norm_type: Normalization type ("unit" for L2 unit norm, "std" for per-dimension std)
        cf_normalization: OLMES CF normalisation scheme ("none", "token", "character", "pmi")
        layer_name_pattern: Pattern for layer names (use {layer_idx} placeholder)
        resume: If True, skip layers that already have a summary.csv on disk
        start_layer: If set, skip all layers before this index

    Returns:
        Dictionary with layer results and summary statistics
    """
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_with_hooks = ModelWithHooks(model)
    scorer = ContinuationProbability(forward_fn=model_with_hooks, tokenizer=tokenizer)

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))
    if start_layer is not None:
        layers = [l for l in layers if l >= start_layer]

    model_name = getattr(model.config, "_name_or_path", "unknown")

    # Pre-compute PMI baselines once (un-steered model, before any layer loop)
    pmi_baselines: Optional[Dict[int, Dict[str, float]]] = None
    if cf_normalization == "pmi":
        print("\nPre-computing PMI baselines (Answer: context)...")
        pmi_context = "Answer: "
        pmi_baselines = {}
        for i, row in ml_test_df.iterrows():
            rr = formatter.format_cf(row)
            pmi_baselines[i] = {
                "target": scorer.continuation_logprob(pmi_context, rr.target).sum_logprob,
                "false1": scorer.continuation_logprob(pmi_context, rr.false1).sum_logprob,
                "false2": scorer.continuation_logprob(pmi_context, rr.false2).sum_logprob,
                "false3": scorer.continuation_logprob(pmi_context, rr.false3).sum_logprob,
            }
        print(f"  PMI baselines computed for {len(pmi_baselines)} questions.")

    all_results = {}
    layer_summaries = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print("=" * 60)

        # Check if this layer was already completed (resume support)
        # Look for summary.csv directly in layer dir or inside a timestamp subdir
        existing_summary = _find_summary_csv(out_path / f"layer_{layer_idx}")
        if resume and existing_summary is not None:
            print(f"Layer {layer_idx} already completed, loading from disk. Skipping.")
            layer_summary = pd.read_csv(existing_summary)
            layer_summary["layer"] = layer_idx
            layer_summaries.append(layer_summary)
            continue

        # Construct layer name using pattern
        layer_name = layer_name_pattern.format(layer_idx=layer_idx)

        # Create DifferenceInMeansSteering for this layer
        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=token_position,
        )

        # Capture activations and compute steering vector
        print(f"\nCapturing activations for layer {layer_idx}...")
        dim_steerer.capture_positive_activations(positive_prompts)
        dim_steerer.capture_negative_activations(negative_prompts)

        steering_vector = dim_steerer.compute_steering_vector(
            normalize=normalize_vector,
            norm_type=norm_type,
        )

        metadata = ExperimentMetadata(
            experiment_name=f"layer_sweep_layer_{layer_idx}",
            model_name=model_name,
            steering_vector_source="difference_in_means",
            layer=layer_idx,
            dataset_name="ml_test",
            dataset_size=len(ml_test_df),
            cf_normalization=cf_normalization,
            notes=f"cf_normalization={cf_normalization}, token_position={token_position}",
        )

        logger = run_cf_experiment(
            scorer=scorer,
            ml_test=ml_test_df,
            dim_steerer=dim_steerer,
            steering_vector=steering_vector,
            coef_list=coef_list,
            output_dir=str(out_path / f"layer_{layer_idx}"),
            formatter=formatter,
            cf_normalization=cf_normalization,
            metadata=metadata,
            pmi_baselines=pmi_baselines,
            verbose_every=verbose_every,
        )

        all_results[layer_idx] = logger

        # Extract summary for this layer
        saved_summary = _find_summary_csv(out_path / f"layer_{layer_idx}")
        if saved_summary is not None:
            layer_summary = pd.read_csv(saved_summary)
            layer_summary["layer"] = layer_idx
            layer_summaries.append(layer_summary)

        # Save combined summary after every layer so progress is not lost
        if layer_summaries:
            combined_summary = pd.concat(layer_summaries, ignore_index=True)
            combined_path = out_path / "combined_summary.csv"
            combined_summary.to_csv(combined_path, index=False)

        print(f"\n>>> Layer {layer_idx} DONE. Results saved to {out_path / f'layer_{layer_idx}'}")

        # Cleanup for next layer
        dim_steerer.cleanup()

    # Final combined summary and plots
    combined_summary = None
    if layer_summaries:
        combined_summary = pd.concat(layer_summaries, ignore_index=True)
        combined_path = out_path / "combined_summary.csv"
        combined_summary.to_csv(combined_path, index=False)
        print(f"\nSaved combined summary to {combined_path}")

        # Generate plots
        _generate_sweep_plots(combined_summary, out_path, cf_normalization)

    return {
        "loggers": all_results,
        "combined_summary": combined_summary,
        "output_dir": out_path,
    }


def _generate_sweep_plots(summary_df: pd.DataFrame, out_dir: Path, cf_normalization: str):
    """Generate plots from sweep results."""
    acc_col = CF_NORM_TO_ACCURACY_COL.get(cf_normalization, "accuracy_char")

    # Plot 1: Accuracy vs coefficient for each layer
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in sorted(summary_df["layer"].unique()):
        layer_data = summary_df[summary_df["layer"] == layer].sort_values("coef")
        ax.plot(layer_data["coef"], layer_data[acc_col], marker="o", label=f"Layer {layer}")

    baseline_acc = summary_df[summary_df["coef"] == 0.0][acc_col].mean()
    ax.axhline(y=baseline_acc, linestyle="--", color="gray", alpha=0.7, label="Baseline")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"CF Accuracy vs Steering Coefficient by Layer ({cf_normalization} norm)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy plot to {out_dir / 'accuracy_by_layer.png'}")

    # Plot 2: Heatmap of accuracy across layers and coefficients
    pivot = summary_df.pivot(index="layer", columns="coef", values=acc_col)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(f"CF Accuracy Heatmap ({cf_normalization} norm)")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {out_dir / 'accuracy_heatmap.png'}")

    # Plot 3: Best coefficient per layer
    best_per_layer = []
    for layer in sorted(summary_df["layer"].unique()):
        layer_data = summary_df[summary_df["layer"] == layer]
        best_row = layer_data.loc[layer_data[acc_col].idxmax()]
        best_per_layer.append({
            "layer": layer,
            "best_coef": best_row["coef"],
            "best_accuracy": best_row[acc_col],
        })
    best_df = pd.DataFrame(best_per_layer)
    best_df.to_csv(out_dir / "best_per_layer.csv", index=False)
    print(f"Saved best coefficients to {out_dir / 'best_per_layer.csv'}")
