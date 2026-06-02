"""
Experiment runner CLI.

Usage:
    python experiments/runner.py configs/my_experiment.yaml
    python experiments/runner.py configs/*.yaml          # run multiple sequentially
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from experiments.config import ExperimentConfig, load_config
from experiments.registry import (
    load_eval_dataset, load_model, load_steering_prompts,
    df_to_fewshot_examples, build_capture_prompts_mcf, build_capture_prompts_cf,
    split_dataset,
)
from experiments import wandb_logger
from olmes.formatter import build_formatter


# =============================================================================
# Core runner
# =============================================================================

def run_experiment(cfg: ExperimentConfig):
    """Run a single experiment defined by cfg."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {cfg.experiment_id}")
    print(f"{'='*70}")

    # --- Setup output directory ---
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a snapshot of the config used
    config_snapshot = out_dir / "config.yaml"
    # Re-read original config path from argv (best effort) or reconstruct
    _save_config_snapshot(cfg, config_snapshot)
    _save_experiment_details(cfg, out_dir / "experiment_details.txt")

    wandb_run = wandb_logger.init_run(cfg)
    try:
        _run_experiment_body(cfg, out_dir, wandb_run)
    finally:
        wandb_logger.finish_run(wandb_run)


def _run_experiment_body(cfg: ExperimentConfig, out_dir, wandb_run):
    # --- Load model ---
    model, tokenizer = load_model(cfg)
    s = cfg.sweep

    # --- Load data and build capture prompts ---
    dataset_capture = bool(cfg.dataset.neg_capture_path)

    if dataset_capture:
        import pandas as pd
        full_eval_df = pd.read_csv(cfg.dataset.eval_path)
        neg_df       = pd.read_csv(cfg.dataset.neg_capture_path)
        false_cols   = [c for c in ["false1", "false2", "false3"] if c in full_eval_df.columns]

        # Few-shots: first num_shots rows of the MMLU general CSV
        fewshot_examples = df_to_fewshot_examples(neg_df, s.num_shots)
        formatter = build_formatter(
            task_prefix=s.task_prefix,
            num_shots=s.num_shots,
            shuffle_choices=s.shuffle_choices,
            seed=s.seed,
            fewshot_examples=fewshot_examples,
        )
        # Zero-shot formatter for DIM capture — no shared context between pos and neg
        capture_formatter = build_formatter(
            task_prefix=s.task_prefix,
            num_shots=0,
            shuffle_choices=s.shuffle_choices,
            seed=s.seed,
        )

        if cfg.dataset.split is not None:
            # --- Random split mode ---
            steering_df, val_df, test_df, steer_idx, val_idx, test_idx = split_dataset(
                full_eval_df, cfg.dataset.split, seed=s.seed,
            )

            # Randomly sample neg_capture from neg pool (after few-shots rows)
            neg_pool = neg_df.iloc[s.num_shots:].reset_index(drop=True)
            n_neg = len(steering_df)
            neg_capture_df = neg_pool.sample(
                n=min(n_neg, len(neg_pool)),
                random_state=s.seed,
            ).reset_index(drop=True)

            # Combine val + test into eval_df with a 'split' column for post-hoc filtering
            val_tagged  = val_df.copy();  val_tagged["split"]  = "validation"
            test_tagged = test_df.copy(); test_tagged["split"] = "test"
            eval_df = pd.concat([val_tagged, test_tagged], ignore_index=True)

            # Save split manifest: eval_question_id (0-based in eval_df) → original_index → split
            manifest_rows = (
                [{"eval_question_id": i, "original_index": orig, "split": "validation"}
                 for i, orig in enumerate(val_idx)]
                + [{"eval_question_id": len(val_df) + i, "original_index": orig, "split": "test"}
                   for i, orig in enumerate(test_idx)]
            )
            pd.DataFrame(manifest_rows).to_csv(out_dir / "split_manifest.csv", index=False)
            print(f"Saved split manifest: {out_dir / 'split_manifest.csv'}")

            pos_df    = steering_df
            capture_n = len(steering_df)
        else:
            # --- Legacy positional mode ---
            capture_n      = cfg.dataset.capture_n
            pos_df         = full_eval_df.iloc[:capture_n]
            neg_capture_df = neg_df.iloc[s.num_shots: s.num_shots + capture_n]
            eval_df        = full_eval_df.iloc[capture_n:].reset_index(drop=True)

        pos_prompts_mcf = build_capture_prompts_mcf(pos_df, capture_formatter, capture_n)
        neg_prompts_mcf = build_capture_prompts_mcf(neg_capture_df, capture_formatter, len(neg_capture_df))
        pos_prompts_cf  = build_capture_prompts_cf(pos_df, capture_formatter, capture_n)
        neg_prompts_cf  = build_capture_prompts_cf(neg_capture_df, capture_formatter, len(neg_capture_df))
    else:
        eval_df, false_cols = load_eval_dataset(cfg)
        pos_prompts, neg_prompts = load_steering_prompts(cfg)
        formatter = build_formatter(
            task_prefix=s.task_prefix,
            num_shots=s.num_shots,
            fewshot_source=s.fewshot_source,
            shuffle_choices=s.shuffle_choices,
            seed=s.seed,
        )
        # Legacy mode: same prompts for both sweep types
        pos_prompts_mcf = pos_prompts_cf = pos_prompts
        neg_prompts_mcf = neg_prompts_cf = neg_prompts

    run_mcf = s.formulation in ("mcf", "both")
    run_cf  = s.formulation in ("cf",  "both")
    has_false_targets = len(false_cols) > 0

    # --- Run summary ---
    _print_run_summary(cfg, s, eval_df, pos_prompts_mcf, neg_prompts_mcf,
                       pos_prompts_cf, neg_prompts_cf, formatter, dataset_capture)

    # --- MCF sweep ---
    if run_mcf:
        print(f"\n>>> Running MCF sweep")
        from single_token_completion_test import sweep_layers_mcf, _compute_summary as _mcf_summary

        mcf_callback = wandb_logger.make_layer_callback(
            wandb_run, prefix="mcf", summarize_fn=_mcf_summary,
        )

        sweep_layers_mcf(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_df,
            positive_prompts=pos_prompts_mcf,
            negative_prompts=neg_prompts_mcf,
            coef_list=s.coef_list,
            formatter=formatter,
            out_dir=str(out_dir / "mcf"),
            layers=s.layers,
            token_position=s.token_position,
            normalize_vector=s.normalize_vector,
            norm_type=s.norm_type,
            layer_name_pattern=s.layer_pattern,
            verbose_every=s.verbose_every,
            resume=s.resume,
            coef_batch_size=s.coef_batch_size,
            capture_batch_size=s.capture_batch_size,
            on_layer_complete=mcf_callback,
        )

    # --- CF sweep ---
    if run_cf:
        if has_false_targets:
            print(f"\n>>> Running CF sweep  (false cols: {false_cols})")
            from token_completion_test import sweep_layers_cf

            # CF summary.csv is already aggregated per coef → identity summarize_fn
            cf_callback = wandb_logger.make_layer_callback(
                wandb_run, prefix="cf", summarize_fn=lambda df: df,
            )

            sweep_layers_cf(
                model=model,
                tokenizer=tokenizer,
                ml_test_df=eval_df,
                positive_prompts=pos_prompts_cf,
                negative_prompts=neg_prompts_cf,
                coef_list=s.coef_list,
                formatter=formatter,
                cf_normalization=s.cf_normalization,
                out_dir=str(out_dir / "cf"),
                layers=s.layers,
                token_position=s.token_position,
                normalize_vector=s.normalize_vector,
                norm_type=s.norm_type,
                layer_name_pattern=s.layer_pattern,
                verbose_every=s.verbose_every,
                resume=s.resume,
                coef_batch_size=s.coef_batch_size,
                capture_batch_size=s.capture_batch_size,
                on_layer_complete=cf_callback,
            )
        else:
            print(f"\n>>> Running CF sweep  (target only, no false cols)")
            _run_cf_target_only(
                model=model,
                tokenizer=tokenizer,
                eval_df=eval_df,
                pos_prompts=pos_prompts_cf,
                neg_prompts=neg_prompts_cf,
                formatter=formatter,
                cfg=cfg,
                out_dir=out_dir / "cf",
                wandb_run=wandb_run,
            )

    # --- Final W&B summary (tables + overview plot) ---
    wandb_logger.log_final_summary(wandb_run, cfg, eval_df, out_dir)

    # --- Qualitative examples ---
    if cfg.sweep.generate_examples:
        print(f"\n>>> Generating qualitative examples")
        _generate_examples_sweep(
            model=model,
            tokenizer=tokenizer,
            eval_df=eval_df,
            positive_prompts=pos_prompts_mcf,
            negative_prompts=neg_prompts_mcf,
            formatter=formatter,
            cfg=cfg,
            out_dir=out_dir / "examples",
        )

    print(f"\n{'='*70}")
    print(f"DONE: {cfg.experiment_id}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*70}\n")


# =============================================================================
# Run summary
# =============================================================================

def _print_run_summary(cfg, s, eval_df, pos_prompts_mcf, neg_prompts_mcf,
                       pos_prompts_cf, neg_prompts_cf, formatter, dataset_capture):
    W = 70
    print(f"\n{'='*W}")
    print(f"  RUN SUMMARY")
    print(f"{'='*W}")

    # Model
    print(f"\n  MODEL")
    print(f"    name    : {cfg.model.name}")
    print(f"    dtype   : {cfg.model.dtype}   device: {cfg.model.device}")

    # Steering target
    print(f"\n  STEERING TARGET")
    print(f"    layer pattern : {s.layer_name_pattern}")
    sublayer_str = s.sublayer if s.sublayer else "(full block output)"
    print(f"    sublayer      : {sublayer_str}")
    print(f"    full pattern  : {s.layer_pattern}")
    layers_str = str(s.layers) if s.layers else "all layers"
    print(f"    layers        : {layers_str}")
    print(f"    token pos     : {s.token_position}   normalize: {s.normalize_vector}"
          + (f"  norm_type: {s.norm_type}" if s.normalize_vector else ""))

    # Sweep
    print(f"\n  SWEEP")
    print(f"    formulation   : {s.formulation}")
    print(f"    coefs         : {s.coef_list}")
    print(f"    eval rows     : {len(eval_df)}")
    if dataset_capture and cfg.dataset.split is not None:
        sp = cfg.dataset.split
        n_val  = sum(1 for _ in eval_df[eval_df["split"] == "validation"].itertuples()) if "split" in eval_df.columns else "?"
        n_test = sum(1 for _ in eval_df[eval_df["split"] == "test"].itertuples()) if "split" in eval_df.columns else "?"
        print(f"      validation  : {n_val} rows ({sp.validation}%)")
        print(f"      test        : {n_test} rows ({sp.test}%)")

    # Capture / steering prompts
    print(f"\n  STEERING PROMPTS  ({'dataset capture' if dataset_capture else 'text-file mode'})")
    if dataset_capture:
        if cfg.dataset.split is not None:
            sp = cfg.dataset.split
            print(f"    mode          : random split (seed={s.seed})")
            print(f"    pos source    : {cfg.dataset.eval_path}  {sp.steering}% randomly sampled")
            print(f"    neg source    : {cfg.dataset.neg_capture_path}  {sp.steering}% randomly sampled (after {s.num_shots} few-shot rows)")
            print(f"    fewshot src   : {cfg.dataset.neg_capture_path}  rows 0–{s.num_shots - 1}")
        else:
            print(f"    mode          : legacy positional (capture_n={cfg.dataset.capture_n})")
            print(f"    pos source    : {cfg.dataset.eval_path}  rows 0–{cfg.dataset.capture_n - 1}")
            print(f"    neg source    : {cfg.dataset.neg_capture_path}"
                  f"  rows {s.num_shots}–{s.num_shots + cfg.dataset.capture_n - 1}")
            print(f"    fewshot src   : {cfg.dataset.neg_capture_path}  rows 0–{s.num_shots - 1}")
    else:
        print(f"    pos file      : {cfg.dataset.positive_prompts_path}"
              f"  ({len(pos_prompts_mcf)} prompts)")
        print(f"    neg file      : {cfg.dataset.negative_prompts_path}"
              f"  ({len(neg_prompts_mcf)} prompts)")
    print(f"    pos count     : {len(pos_prompts_mcf)} MCF  /  {len(pos_prompts_cf)} CF")
    print(f"    neg count     : {len(neg_prompts_mcf)} MCF  /  {len(neg_prompts_cf)} CF")
    print(f"    few-shots     : {s.num_shots}")

    def _show(label, text):
        print(f"\n  {label}:")
        print(f"  {'-'*66}")
        for line in text.strip().splitlines():
            print(f"    {line}")

    # All few-shot blocks — extracted from the eval formatter, not capture prompts
    row0 = eval_df.iloc[0]
    mcf_row0 = formatter.format_mcf(row0, question_idx=0)
    eval_blocks = mcf_row0.prompt.split("\n\n")
    fewshot_blocks = eval_blocks[:-1] if len(eval_blocks) > 1 else []
    if fewshot_blocks:
        for i, block in enumerate(fewshot_blocks):
            _show(f"FEW-SHOT {i + 1} of {len(fewshot_blocks)}", block)
    else:
        _show("FEW-SHOTS", "(none — zero-shot)")

    # DIM capture prompts — what the model actually sees when building the steering vector
    capture_mode_label = "MCF-formatted dataset capture" if dataset_capture else "plain text (text-file mode)"
    print(f"\n  DIM CAPTURE FORMAT: {capture_mode_label}")
    _show("POS capture prompt (used to build steering vector)", pos_prompts_mcf[0])
    _show("NEG capture prompt (used to build steering vector)", neg_prompts_mcf[0])
    if dataset_capture and pos_prompts_mcf is not pos_prompts_cf:
        _show("POS capture prompt CF variant", pos_prompts_cf[0])
        _show("NEG capture prompt CF variant", neg_prompts_cf[0])

    # Eval question: a question from the test set (rows after the capture split)
    # This is what the model is actually scored on — steering is never applied during capture
    row0 = eval_df.iloc[0]
    _show("EXAMPLE eval question (raw)", f"Q : {row0['prompt']}\nA : {row0['target']}")

    # Show how the eval question looks when formatted for each formulation
    mcf_row = formatter.format_mcf(row0, question_idx=0)
    _show("EXAMPLE eval question as MCF input (what model sees)",
          mcf_row.prompt.split("\n\n")[-1] + f"\n[correct label: {mcf_row.correct_label}]")

    cf_row = formatter.format_cf(row0)
    _show("EXAMPLE eval question as CF input (what model sees)",
          cf_row.prompt.split("\n\n")[-1] + f"\n[target continuation: '{cf_row.target.strip()}']")

    print(f"\n{'='*W}\n")


# =============================================================================
# Qualitative example generation
# =============================================================================

def _generate_examples_sweep(
    model,
    tokenizer,
    eval_df,
    positive_prompts,
    negative_prompts,
    formatter,
    cfg,
    out_dir,
):
    """
    For each layer and coef, generate free-text completions for n_examples
    questions. Saves results/exp_id/examples/examples.csv with columns:
        layer, coef, question_id, prompt, target, generated
    """
    import random
    import pandas as pd
    import torch
    from pathlib import Path
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering

    s = cfg.sweep
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    out_file = out_path / "examples.csv"
    if s.resume and out_file.exists():
        print(f"Examples already generated, skipping ({out_file})")
        return out_file

    model_with_hooks = ModelWithHooks(model)
    layers = s.layers if s.layers else list(range(model.config.num_hidden_layers))

    rng = random.Random(s.seed)
    sample_indices = rng.sample(range(len(eval_df)), min(s.n_examples, len(eval_df)))
    coefs = [0.0] + [c for c in s.coef_list if c != 0.0]

    all_records = []

    for layer_idx in layers:
        print(f"  [examples] Layer {layer_idx}")
        layer_name = s.layer_pattern.format(layer_idx=layer_idx)

        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=s.token_position,
            capture_batch_size=s.capture_batch_size,
        )
        dim_steerer.capture_positive_activations(positive_prompts)
        dim_steerer.capture_negative_activations(negative_prompts)
        steering_vector = dim_steerer.compute_steering_vector(
            normalize=s.normalize_vector,
            norm_type=s.norm_type,
        )

        for coef in coefs:
            if coef != 0.0:
                dim_steerer.apply_steering(steering_vector, coefficient=coef)
            try:
                for idx in sample_indices:
                    row = eval_df.iloc[idx]
                    formatted = formatter.format_cf(row)
                    prompt = formatted.prompt

                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=s.max_length,
                    ).to(next(model_with_hooks.model.parameters()).device)

                    with torch.no_grad():
                        output_ids = model_with_hooks.model.generate(
                            **inputs,
                            max_new_tokens=s.max_new_tokens,
                            do_sample=False,
                        )

                    generated = tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                    all_records.append({
                        "layer": layer_idx,
                        "coef": coef,
                        "question_id": idx,
                        "prompt": prompt,
                        "target": str(row["target"]),
                        "generated": generated,
                    })
            finally:
                if coef != 0.0:
                    dim_steerer.reset_steering()

        dim_steerer.cleanup()

    df = pd.DataFrame(all_records)
    df.to_csv(out_file, index=False)
    print(f"  Saved {len(df)} example generations to {out_file}")
    return out_file


# =============================================================================
# Continuation scorer for target-only datasets (no false columns)
# =============================================================================

def _run_cf_target_only(model, tokenizer, eval_df, pos_prompts,
                        neg_prompts, formatter, cfg, out_dir, wandb_run=None):
    """
    Run continuation scoring when there are no false targets.
    Logs logprob of the full target string per (layer, question, coef).
    """
    import pandas as pd
    import torch
    from pathlib import Path
    from dataclasses import asdict

    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering
    from token_completion_test import ContinuationProbability

    s = cfg.sweep
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_with_hooks = ModelWithHooks(model)
    scorer = ContinuationProbability(
        forward_fn=model_with_hooks,
        tokenizer=tokenizer,
        max_length=s.max_length,
    )

    layers = s.layers
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    coefs = [0.0] + [c for c in s.coef_list if c != 0.0]

    all_records = []
    baselines = {}  # (layer, question_id) -> logprob at coef=0

    for layer_idx in layers:
        layer_result_path = out_path / f"layer_{layer_idx}_results.csv"

        if s.resume and layer_result_path.exists():
            print(f"Layer {layer_idx}: already done, loading from disk.")
            df = pd.read_csv(layer_result_path)
            all_records.append(df)
            for _, r in df[df["coef"] == 0.0].iterrows():
                baselines[(int(r["layer"]), int(r["question_id"]))] = float(r["sum_logprob"])
            continue

        print(f"\n{'='*60}\nLayer {layer_idx}\n{'='*60}")
        layer_name = s.layer_pattern.format(layer_idx=layer_idx)

        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=s.token_position,
            capture_batch_size=s.capture_batch_size,
        )
        dim_steerer.capture_positive_activations(pos_prompts)
        dim_steerer.capture_negative_activations(neg_prompts)
        steering_vector = dim_steerer.compute_steering_vector(
            normalize=s.normalize_vector,
            norm_type=s.norm_type,
        )

        layer_records = []

        for coef in coefs:
            print(f"\n  [Layer {layer_idx}] coef={coef}")
            if coef != 0.0:
                dim_steerer.apply_steering(steering_vector, coefficient=coef)
            try:
                for i, row in eval_df.iterrows():
                    rr = formatter.format_cf(row)
                    result = scorer.continuation_logprob(rr.prompt, rr.target)

                    key = (layer_idx, i)
                    if coef == 0.0:
                        baselines[key] = result.sum_logprob

                    base_lp = baselines.get(key, result.sum_logprob)
                    delta = result.sum_logprob - base_lp if coef != 0.0 else 0.0

                    layer_records.append({
                        "layer": layer_idx,
                        "question_id": i,
                        "coef": coef,
                        "prompt": rr.prompt,
                        "target_text": rr.target,
                        "token_count": result.token_count,
                        "sum_logprob": result.sum_logprob,
                        "mean_logprob": result.mean_logprob,
                        "char_norm_logprob": result.char_norm_logprob,
                        "delta_sum_logprob": delta,
                    })

                    if s.verbose_every and (i + 1) % s.verbose_every == 0:
                        print(f"    Processed {i + 1}/{len(eval_df)}")
            finally:
                if coef != 0.0:
                    dim_steerer.reset_steering()

        layer_df = pd.DataFrame(layer_records)
        layer_df.to_csv(layer_result_path, index=False)
        print(f"\n>>> Layer {layer_idx} COMPLETE — {len(layer_df)} records")

        all_records.append(layer_df)

        # Scan disk so prior-session layers still show up in combined outputs
        combined = _concat_layer_csvs(out_path)
        if combined is not None:
            combined.to_csv(out_path / "combined_results.csv", index=False)
            _save_continuation_summary(combined, out_path / "combined_summary.csv")

        if wandb_run is not None:
            try:
                summary_path = out_path / "combined_summary.csv"
                if summary_path.exists():
                    layer_summary = pd.read_csv(summary_path)
                    layer_summary = layer_summary[layer_summary["layer"] == layer_idx]
                    if len(layer_summary):
                        wandb_logger._log_summary_rows(
                            wandb_run, layer_summary, prefix="cf", step=int(layer_idx),
                        )
            except Exception as e:
                print(f"  wandb log raised: {e}")

        dim_steerer.cleanup()

    combined = _concat_layer_csvs(out_path)
    if combined is not None:
        combined.to_csv(out_path / "combined_results.csv", index=False)
        _save_continuation_summary(combined, out_path / "combined_summary.csv")
        print(f"Saved combined results to {out_path} ({combined['layer'].nunique()} layers)")


def _concat_layer_csvs(out_path: Path):
    """Concat every layer_*_results.csv under out_path. Returns None if none exist."""
    import pandas as pd
    paths = sorted(out_path.glob("layer_*_results.csv"))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _save_continuation_summary(df: pd.DataFrame, path):
    rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row = {
            "layer": int(layer),
            "coef": float(coef),
            "n": len(grp),
            "mean_sum_logprob": grp["sum_logprob"].mean(),
            "mean_mean_logprob": grp["mean_logprob"].mean(),
        }
        if coef != 0.0:
            row["mean_delta_sum_logprob"] = grp["delta_sum_logprob"].mean()
            row["pct_improved"] = (grp["delta_sum_logprob"] > 0).mean()
            row["pct_hurt"] = (grp["delta_sum_logprob"] < 0).mean()
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).sort_values(["layer", "coef"]).to_csv(path, index=False)


# =============================================================================
# Config snapshot
# =============================================================================

def _save_experiment_details(cfg: ExperimentConfig, path: Path):
    """Write a human-readable summary of the experiment setup to a txt file."""
    from datetime import datetime
    s = cfg.sweep
    dataset_capture = bool(cfg.dataset.neg_capture_path)
    lines = [
        f"EXPERIMENT: {cfg.experiment_id}",
        f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "MODEL",
        f"  name    : {cfg.model.name}",
        f"  dtype   : {cfg.model.dtype}   device: {cfg.model.device}",
        "",
        "STEERING TARGET",
        f"  pattern : {s.layer_pattern}",
        f"  sublayer: {s.sublayer or '(full block output)'}",
        f"  layers  : {s.layers or 'all'}",
        f"  token_position : {s.token_position}",
        f"  normalize      : {s.normalize_vector}  norm_type: {s.norm_type}",
        "",
        "SWEEP",
        f"  formulation    : {s.formulation}",
        f"  coefs          : {s.coef_list}",
        f"  cf_normalization: {s.cf_normalization}",
        "",
        "DATASET",
        f"  eval_path      : {cfg.dataset.eval_path}",
        f"  capture_mode   : {'dataset (MCF-formatted)' if dataset_capture else 'text-file (plain sentences)'}",
    ]
    if dataset_capture:
        if cfg.dataset.split is not None:
            sp = cfg.dataset.split
            lines += [
                f"  split_mode     : random (seed={s.seed})",
                f"  split.steering : {sp.steering}%  → pos DIM capture",
                f"  split.validation: {sp.validation}%",
                f"  split.test     : {sp.test}%",
                f"  pos_capture    : {cfg.dataset.eval_path}  (randomly sampled {sp.steering}%)",
                f"  neg_capture    : {cfg.dataset.neg_capture_path}  (randomly sampled {sp.steering}% from rows after few-shots)",
            ]
        else:
            lines += [
                f"  split_mode     : legacy positional (capture_n={cfg.dataset.capture_n})",
                f"  pos_capture    : {cfg.dataset.eval_path}  rows 0–{cfg.dataset.capture_n - 1}",
                f"  neg_capture    : {cfg.dataset.neg_capture_path}  rows {s.num_shots}–{s.num_shots + cfg.dataset.capture_n - 1}",
            ]
    else:
        lines += [
            f"  pos_prompts    : {cfg.dataset.positive_prompts_path}",
            f"  neg_prompts    : {cfg.dataset.negative_prompts_path}",
        ]
    lines += [
        f"  few_shots      : {s.num_shots}  source: {s.fewshot_source or 'none'}",
        f"  shuffle_choices: {s.shuffle_choices}",
        "",
        "OUTPUT",
        f"  base_dir: {cfg.output.base_dir}",
        f"  out_dir : {cfg.output_dir}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _save_config_snapshot(cfg: ExperimentConfig, path: Path):
    """Save a YAML snapshot of the config used for this run."""
    import dataclasses
    snapshot = dataclasses.asdict(cfg)
    with open(path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# CLI
# =============================================================================

def _parse_layers_arg(tokens):
    """Parse --layers tokens into a sorted de-duplicated list of ints.

    Accepts individual ints and ranges with a hyphen, e.g.:
        --layers 0 1 2        -> [0, 1, 2]
        --layers 0-5          -> [0, 1, 2, 3, 4, 5]
        --layers 0-3 7 10-11  -> [0, 1, 2, 3, 7, 10, 11]
    """
    out: set[int] = set()
    for tok in tokens:
        tok = tok.strip()
        if "-" in tok and not tok.startswith("-"):
            start_s, end_s = tok.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {tok}")
            out.update(range(start, end + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(
        description="Run steering vector experiments from YAML config files."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Path(s) to YAML experiment config file(s)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Override sweep.layers. Accepts ints and ranges, e.g. '0 1 2' or '0-5 7'",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Override output.base_dir (e.g. /content/drive/MyDrive/steering_results)",
    )
    parser.add_argument(
        "--examples", dest="generate_examples", action="store_true", default=None,
        help="Override: enable qualitative example generation",
    )
    parser.add_argument(
        "--no-examples", dest="generate_examples", action="store_false",
        help="Override: disable qualitative example generation",
    )
    args = parser.parse_args()

    configs = []
    for pattern in args.configs:
        matched = list(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        configs.extend(matched)

    if not configs:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    layers_override = _parse_layers_arg(args.layers) if args.layers else None

    print(f"Running {len(configs)} experiment(s): {[c.name for c in configs]}")
    if layers_override is not None:
        print(f"  --layers override: {layers_override}")
    if args.base_dir is not None:
        print(f"  --base-dir override: {args.base_dir}")

    for config_path in configs:
        cfg = load_config(str(config_path))
        if layers_override is not None:
            cfg.sweep.layers = layers_override
        if args.base_dir is not None:
            cfg.output.base_dir = args.base_dir
        if args.generate_examples is not None:
            cfg.sweep.generate_examples = args.generate_examples
        run_experiment(cfg)


if __name__ == "__main__":
    main()
