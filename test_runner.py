"""
Tests for the random split pipeline changes.

Run in Colab with:
    !python test_runner.py

Tests cover:
  - SplitConfig parsing from YAML dict
  - split_dataset: correct sizes, no overlap, reproducibility, seed sensitivity
  - Config validation: percentages must sum to 100, each > 0, neg_capture_path required
  - load_config round-trip with split section present / absent
  - seed field in SweepConfig
  - Backward compat: no split → legacy positional path still intact
"""

import sys
import os
import tempfile
import textwrap
import traceback
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import torch

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        _results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        _results.append((name, False, traceback.format_exc()))


# =============================================================================
# Helpers
# =============================================================================

def _make_eval_csv(tmp_dir, n=100, with_false=True):
    rows = []
    for i in range(n):
        r = {"prompt": f"Q{i}", "target": f"A{i}"}
        if with_false:
            r.update({"false1": f"B{i}", "false2": f"C{i}", "false3": f"D{i}"})
        rows.append(r)
    path = tmp_dir / "eval.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _make_neg_csv(tmp_dir, n=50, with_false=True):
    rows = []
    for i in range(n):
        r = {"prompt": f"NQ{i}", "target": f"NA{i}"}
        if with_false:
            r.update({"false1": f"NB{i}", "false2": f"NC{i}", "false3": f"ND{i}"})
        rows.append(r)
    path = tmp_dir / "neg.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _make_config_yaml(tmp_dir, extra_dataset="", extra_sweep=""):
    eval_path = _make_eval_csv(tmp_dir)
    neg_path  = _make_neg_csv(tmp_dir)
    yaml_text = textwrap.dedent(f"""
        experiment_id: test_exp

        model:
          name: meta-llama/Meta-Llama-3-8B
          dtype: float16
          device: cpu

        dataset:
          eval_path: {eval_path}
          neg_capture_path: {neg_path}
          split:
            steering: 20
            validation: 20
            test: 60
        {extra_dataset}

        sweep:
          seed: 42
          formulation: mcf
          num_shots: 5
        {extra_sweep}

        output:
          base_dir: /tmp/test_results/
    """)
    cfg_path = tmp_dir / "test.yaml"
    cfg_path.write_text(yaml_text)
    return str(cfg_path)


# =============================================================================
# 1. SplitConfig dataclass
# =============================================================================

def _test_split_config_defaults():
    from experiments.config import SplitConfig
    sp = SplitConfig()
    assert sp.steering == 20
    assert sp.validation == 20
    assert sp.test == 60

def _test_split_config_custom():
    from experiments.config import SplitConfig
    sp = SplitConfig(steering=10, validation=30, test=60)
    assert sp.steering == 10
    assert sp.validation == 30
    assert sp.test == 60


# =============================================================================
# 2. split_dataset function
# =============================================================================

def _make_df(n=100):
    return pd.DataFrame({"prompt": [f"Q{i}" for i in range(n)], "target": [f"A{i}" for i in range(n)]})


def _test_split_sizes_exact():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    s_df, v_df, t_df, si, vi, ti = split_dataset(df, sp, seed=42)
    assert len(s_df) == 20, f"steering size {len(s_df)}"
    assert len(v_df) == 20, f"val size {len(v_df)}"
    assert len(t_df) == 60, f"test size {len(t_df)}"


def _test_split_sizes_cover_all():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    s_df, v_df, t_df, si, vi, ti = split_dataset(df, sp, seed=42)
    all_idx = sorted(si + vi + ti)
    assert all_idx == list(range(100)), "indices don't cover full dataset"


def _test_split_no_overlap():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, _, _, si, vi, ti = split_dataset(df, sp, seed=42)
    assert len(set(si) & set(vi)) == 0, "steering/val overlap"
    assert len(set(si) & set(ti)) == 0, "steering/test overlap"
    assert len(set(vi) & set(ti)) == 0, "val/test overlap"


def _test_split_reproducibility():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, _, _, si1, vi1, ti1 = split_dataset(df, sp, seed=42)
    _, _, _, si2, vi2, ti2 = split_dataset(df, sp, seed=42)
    assert si1 == si2 and vi1 == vi2 and ti1 == ti2, "same seed gave different split"


def _test_split_different_seeds():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, _, _, si1, vi1, ti1 = split_dataset(df, sp, seed=42)
    _, _, _, si2, vi2, ti2 = split_dataset(df, sp, seed=99)
    assert si1 != si2, "different seeds gave identical steering split"


def _test_split_dataframe_values_match():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    s_df, _, _, si, _, _ = split_dataset(df, sp, seed=42)
    for i, orig_i in enumerate(si):
        assert s_df.iloc[i]["prompt"] == df.iloc[orig_i]["prompt"], \
            f"steering_df row {i} doesn't match original index {orig_i}"


def _test_split_rounding_on_odd_n():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(97)  # not divisible cleanly
    sp = SplitConfig(steering=20, validation=20, test=60)
    s_df, v_df, t_df, si, vi, ti = split_dataset(df, sp, seed=42)
    assert len(si) + len(vi) + len(ti) == 97, "rows lost in rounding"
    assert len(set(si) | set(vi) | set(ti)) == 97, "duplicate rows across splits"


# =============================================================================
# 3. Config loading with split
# =============================================================================

def _test_load_config_with_split():
    from experiments.config import load_config, SplitConfig
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config_yaml(Path(tmp))
        cfg = load_config(cfg_path)
        assert cfg.dataset.split is not None
        assert isinstance(cfg.dataset.split, SplitConfig)
        assert cfg.dataset.split.steering == 20
        assert cfg.dataset.split.validation == 20
        assert cfg.dataset.split.test == 60


def _test_load_config_seed_field():
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config_yaml(Path(tmp))
        cfg = load_config(cfg_path)
        assert cfg.sweep.seed == 42


def _test_load_config_seed_default():
    """seed defaults to 42 when not specified in YAML."""
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: test_noseed
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              split:
                steering: 20
                validation: 20
                test: 60
            sweep:
              formulation: mcf
              num_shots: 5
            output:
              base_dir: /tmp/test_results/
        """)
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_config(str(cfg_path))
        assert cfg.sweep.seed == 42, f"expected default seed 42, got {cfg.sweep.seed}"


def _test_load_config_without_split():
    """Configs without split: should still load and have split=None."""
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: test_nosplit
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              capture_n: 20
            sweep:
              formulation: mcf
              num_shots: 5
            output:
              base_dir: /tmp/test_results/
        """)
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_config(str(cfg_path))
        assert cfg.dataset.split is None
        assert cfg.dataset.capture_n == 20


# =============================================================================
# 4. Validation errors
# =============================================================================

def _load_bad_config(tmp_dir, yaml_text):
    """Returns (cfg, error_text) — error_text is non-empty if sys.exit was called."""
    import io
    from unittest.mock import patch
    cfg_path = Path(tmp_dir) / "bad.yaml"
    cfg_path.write_text(yaml_text)

    captured_stderr = io.StringIO()
    exit_called = []

    def fake_exit(code=0):
        exit_called.append(code)
        raise SystemExit(code)

    from experiments import config as cfg_mod
    with patch.object(sys, "stderr", captured_stderr), \
         patch.object(sys, "exit", fake_exit):
        try:
            from experiments.config import load_config
            cfg = load_config(str(cfg_path))
            return cfg, ""
        except SystemExit:
            return None, captured_stderr.getvalue()


def _test_validation_split_sum_not_100():
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: bad
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              split:
                steering: 20
                validation: 20
                test: 50
            sweep:
              formulation: mcf
            output:
              base_dir: /tmp/
        """)
        cfg, err = _load_bad_config(Path(tmp), yaml_text)
        assert cfg is None, "expected validation to fail"
        assert "sum" in err or "100" in err, f"expected sum error, got: {err!r}"


def _test_validation_split_zero_value():
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: bad
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              split:
                steering: 0
                validation: 40
                test: 60
            sweep:
              formulation: mcf
            output:
              base_dir: /tmp/
        """)
        cfg, err = _load_bad_config(Path(tmp), yaml_text)
        assert cfg is None, "expected validation to fail for zero steering"
        assert "steering" in err or "> 0" in err, f"expected >0 error, got: {err!r}"


def _test_validation_split_requires_neg_capture_path():
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: bad
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              split:
                steering: 20
                validation: 20
                test: 60
            sweep:
              formulation: mcf
            output:
              base_dir: /tmp/
        """)
        cfg, err = _load_bad_config(Path(tmp), yaml_text)
        assert cfg is None, "expected validation to fail (missing neg_capture_path)"
        assert "neg_capture_path" in err, f"expected neg_capture_path error, got: {err!r}"


# =============================================================================
# 5. Split manifest content (unit-test the runner logic without model)
# =============================================================================

def _test_manifest_eval_ids_are_contiguous():
    """val rows get ids 0..n_val-1, test rows get n_val..n_val+n_test-1."""
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, val_df, test_df, _, val_idx, test_idx = split_dataset(df, sp, seed=42)

    manifest_rows = (
        [{"eval_question_id": i, "original_index": orig, "split": "validation"}
         for i, orig in enumerate(val_idx)]
        + [{"eval_question_id": len(val_df) + i, "original_index": orig, "split": "test"}
           for i, orig in enumerate(test_idx)]
    )
    manifest = pd.DataFrame(manifest_rows)

    val_ids  = manifest[manifest["split"] == "validation"]["eval_question_id"].tolist()
    test_ids = manifest[manifest["split"] == "test"]["eval_question_id"].tolist()

    assert val_ids == list(range(20)), f"val ids: {val_ids}"
    assert test_ids == list(range(20, 80)), f"test ids: {test_ids}"


def _test_manifest_original_indices_no_overlap():
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, val_df, test_df, steer_idx, val_idx, test_idx = split_dataset(df, sp, seed=42)
    all_tracked = set(steer_idx) | set(val_idx) | set(test_idx)
    assert len(all_tracked) == 100


def _test_eval_df_split_column():
    """eval_df built from val+test tagged should have correct split values."""
    from experiments.config import SplitConfig
    from experiments.registry import split_dataset
    df = _make_df(100)
    sp = SplitConfig(steering=20, validation=20, test=60)
    _, val_df, test_df, _, _, _ = split_dataset(df, sp, seed=42)
    val_tagged  = val_df.copy();  val_tagged["split"]  = "validation"
    test_tagged = test_df.copy(); test_tagged["split"] = "test"
    eval_df = pd.concat([val_tagged, test_tagged], ignore_index=True)

    assert len(eval_df) == 80
    assert (eval_df["split"] == "validation").sum() == 20
    assert (eval_df["split"] == "test").sum() == 60
    assert list(eval_df["split"].unique()) == ["validation", "test"] or \
           set(eval_df["split"].unique()) == {"validation", "test"}


# =============================================================================
# 6. Neg capture sampling
# =============================================================================

def _test_neg_capture_excludes_fewshot_rows():
    """Neg capture should never include the first num_shots rows of neg_df."""
    neg_df = pd.DataFrame({"prompt": [f"NQ{i}" for i in range(50)],
                           "target": [f"NA{i}" for i in range(50)],
                           "false1": ["x"]*50, "false2": ["y"]*50, "false3": ["z"]*50})
    num_shots = 5
    seed = 42
    n_neg = 20
    neg_pool = neg_df.iloc[num_shots:].reset_index(drop=True)
    sampled = neg_pool.sample(n=min(n_neg, len(neg_pool)), random_state=seed).reset_index(drop=True)

    fewshot_prompts = set(neg_df.iloc[:num_shots]["prompt"])
    sampled_prompts = set(sampled["prompt"])
    assert len(fewshot_prompts & sampled_prompts) == 0, \
        f"few-shot rows leaked into neg capture: {fewshot_prompts & sampled_prompts}"


def _test_neg_capture_reproducible():
    neg_df = pd.DataFrame({"prompt": [f"NQ{i}" for i in range(50)],
                           "target": [f"NA{i}" for i in range(50)]})
    pool = neg_df.iloc[5:].reset_index(drop=True)
    s1 = pool.sample(n=20, random_state=42)["prompt"].tolist()
    s2 = pool.sample(n=20, random_state=42)["prompt"].tolist()
    assert s1 == s2, "same seed gave different neg samples"


def _test_neg_capture_different_seed():
    neg_df = pd.DataFrame({"prompt": [f"NQ{i}" for i in range(50)],
                           "target": [f"NA{i}" for i in range(50)]})
    pool = neg_df.iloc[5:].reset_index(drop=True)
    s1 = pool.sample(n=20, random_state=42)["prompt"].tolist()
    s2 = pool.sample(n=20, random_state=99)["prompt"].tolist()
    assert s1 != s2, "different seeds gave identical neg samples"


# =============================================================================
# 7. W&B logging data pipeline (no W&B account or GPU needed)
# =============================================================================

def _make_eval_df_with_split(n_val=2, n_test=4):
    val  = pd.DataFrame({"prompt": [f"Q{i}"   for i in range(n_val)],
                         "target": [f"A{i}"   for i in range(n_val)],
                         "split":  ["validation"] * n_val})
    test = pd.DataFrame({"prompt": [f"Q{n_val+i}" for i in range(n_test)],
                         "target": [f"A{n_val+i}" for i in range(n_test)],
                         "split":  ["test"] * n_test})
    return pd.concat([val, test], ignore_index=True)


def _test_compute_split_summary_acc_and_delta():
    from experiments.wandb_logger import _compute_split_summary
    df = pd.DataFrame([
        {"layer": 10, "coef": 0.0, "correct": 1, "delta_correct_logprob": 0.0},
        {"layer": 10, "coef": 0.0, "correct": 0, "delta_correct_logprob": 0.0},
        {"layer": 10, "coef": 5.0, "correct": 1, "delta_correct_logprob": 0.1},
        {"layer": 10, "coef": 5.0, "correct": 1, "delta_correct_logprob": 0.3},
    ])
    s = _compute_split_summary(df, "correct", "delta_correct_logprob")
    base    = s[(s["layer"] == 10) & (s["coef"] == 0.0)].iloc[0]
    steered = s[(s["layer"] == 10) & (s["coef"] == 5.0)].iloc[0]
    assert base["acc"] == 0.5,          f"baseline acc {base['acc']}"
    assert base["delta_mean"] == 0.0,   "baseline delta must be 0"
    assert steered["acc"] == 1.0,       f"steered acc {steered['acc']}"
    assert abs(steered["delta_mean"] - 0.2) < 1e-9, f"delta_mean {steered['delta_mean']}"


def _test_compute_split_summary_n():
    from experiments.wandb_logger import _compute_split_summary
    df = pd.DataFrame([
        {"layer": 10, "coef": 5.0, "correct": 1, "delta_correct_logprob": 0.1},
        {"layer": 10, "coef": 5.0, "correct": 0, "delta_correct_logprob": -0.1},
        {"layer": 10, "coef": 5.0, "correct": 1, "delta_correct_logprob": 0.05},
    ])
    s = _compute_split_summary(df, "correct", "delta_correct_logprob")
    assert s.iloc[0]["n"] == 3


def _test_select_best_coef_val_acc():
    from experiments.wandb_logger import _select_best_coef
    val_sum = pd.DataFrame([
        {"layer": 10, "coef":  0.0, "acc": 0.5, "delta_mean":  0.00},
        {"layer": 10, "coef":  5.0, "acc": 0.8, "delta_mean":  0.10},
        {"layer": 10, "coef": -5.0, "acc": 0.4, "delta_mean": -0.10},
    ])
    r = _select_best_coef(val_sum, "val_acc")
    assert r.iloc[0]["best_coef"] == 5.0
    assert r.iloc[0]["val_acc"]   == 0.8


def _test_select_best_coef_val_delta():
    from experiments.wandb_logger import _select_best_coef
    val_sum = pd.DataFrame([
        {"layer": 10, "coef":  0.0, "acc": 0.5, "delta_mean":  0.00},
        {"layer": 10, "coef":  5.0, "acc": 0.6, "delta_mean":  0.30},
        {"layer": 10, "coef": -5.0, "acc": 0.7, "delta_mean": -0.10},
    ])
    r = _select_best_coef(val_sum, "val_delta")
    assert r.iloc[0]["best_coef"] == 5.0, "val_delta should pick coef with highest delta"


def _test_select_best_coef_excludes_baseline():
    from experiments.wandb_logger import _select_best_coef
    val_sum = pd.DataFrame([
        {"layer": 10, "coef": 0.0, "acc": 0.99, "delta_mean": 0.0},  # highest acc but coef=0
        {"layer": 10, "coef": 5.0, "acc": 0.60, "delta_mean": 0.1},
    ])
    r = _select_best_coef(val_sum, "val_acc")
    assert r.iloc[0]["best_coef"] == 5.0, "coef=0 must never be selected as best"


def _test_select_best_coef_tiebreak_on_delta():
    from experiments.wandb_logger import _select_best_coef
    val_sum = pd.DataFrame([
        {"layer": 10, "coef":  5.0, "acc": 0.7, "delta_mean": 0.20},
        {"layer": 10, "coef": 10.0, "acc": 0.7, "delta_mean": 0.05},  # same acc, lower delta
    ])
    r = _select_best_coef(val_sum, "val_acc")
    assert r.iloc[0]["best_coef"] == 5.0, "tiebreak should use delta_mean"


def _test_select_best_coef_independent_per_layer():
    from experiments.wandb_logger import _select_best_coef
    val_sum = pd.DataFrame([
        {"layer": 10, "coef":  5.0, "acc": 0.8, "delta_mean":  0.1},
        {"layer": 10, "coef": -5.0, "acc": 0.4, "delta_mean": -0.1},
        {"layer": 20, "coef":  5.0, "acc": 0.3, "delta_mean":  0.05},
        {"layer": 20, "coef": -5.0, "acc": 0.9, "delta_mean": -0.2},
    ])
    r = _select_best_coef(val_sum, "val_acc")
    assert r[r["layer"] == 10].iloc[0]["best_coef"] ==  5.0
    assert r[r["layer"] == 20].iloc[0]["best_coef"] == -5.0, "each layer picks independently"


def _test_load_mcf_results_attaches_split():
    from experiments.wandb_logger import _load_mcf_results
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            {"layer": 10, "question_id": 0, "coef": 0.0, "correct": 1, "delta_correct_logprob": 0.0},
            {"layer": 10, "question_id": 1, "coef": 0.0, "correct": 0, "delta_correct_logprob": 0.0},
            {"layer": 10, "question_id": 2, "coef": 0.0, "correct": 1, "delta_correct_logprob": 0.0},
        ]
        pd.DataFrame(rows).to_csv(Path(tmp) / "layer_10_results.csv", index=False)
        eval_df = _make_eval_df_with_split(n_val=2, n_test=1)
        result = _load_mcf_results(tmp, eval_df)
        assert result is not None
        assert "split" in result.columns
        assert result[result["question_id"] == 0]["split"].values[0] == "validation"
        assert result[result["question_id"] == 2]["split"].values[0] == "test"


def _test_load_mcf_results_no_split_col():
    from experiments.wandb_logger import _load_mcf_results
    with tempfile.TemporaryDirectory() as tmp:
        rows = [{"layer": 10, "question_id": 0, "coef": 0.0, "correct": 1, "delta_correct_logprob": 0.0}]
        pd.DataFrame(rows).to_csv(Path(tmp) / "layer_10_results.csv", index=False)
        eval_df = pd.DataFrame({"prompt": ["Q0"], "target": ["A"]})  # no split column
        result = _load_mcf_results(tmp, eval_df)
        assert result is not None
        assert "split" not in result.columns, "split should not appear when eval_df has none"


def _test_load_mcf_results_no_files_returns_none():
    from experiments.wandb_logger import _load_mcf_results
    with tempfile.TemporaryDirectory() as tmp:
        result = _load_mcf_results(tmp, pd.DataFrame())
        assert result is None


def _test_load_cf_results_attaches_split():
    from experiments.wandb_logger import _load_cf_results
    with tempfile.TemporaryDirectory() as tmp:
        layer_dir = Path(tmp) / "layer_15"
        layer_dir.mkdir()
        rows = [
            {"question_id": 0, "coef": 0.0, "correct_char": 1, "delta_target_char_norm_lp": 0.0},
            {"question_id": 1, "coef": 0.0, "correct_char": 0, "delta_target_char_norm_lp": 0.0},
        ]
        pd.DataFrame(rows).to_csv(layer_dir / "detailed_wide.csv", index=False)
        eval_df = _make_eval_df_with_split(n_val=1, n_test=1)
        result = _load_cf_results(tmp, eval_df)
        assert result is not None
        assert "layer" in result.columns
        assert result["layer"].iloc[0] == 15
        assert result[result["question_id"] == 0]["split"].values[0] == "validation"
        assert result[result["question_id"] == 1]["split"].values[0] == "test"


def _test_load_cf_results_no_files_returns_none():
    from experiments.wandb_logger import _load_cf_results
    with tempfile.TemporaryDirectory() as tmp:
        result = _load_cf_results(tmp, pd.DataFrame())
        assert result is None


def _test_log_final_summary_run_none_is_noop():
    from experiments.wandb_logger import log_final_summary
    # run=None → must return immediately without touching cfg/eval_df/out_dir
    log_final_summary(None, None, None, None)


# =============================================================================
# 8. ModelConfig: load_in_8bit field
# =============================================================================

def _test_model_config_load_in_8bit_default():
    from experiments.config import ModelConfig
    mc = ModelConfig(name="x")
    assert mc.load_in_8bit is False, f"expected False, got {mc.load_in_8bit}"


def _test_model_config_load_in_8bit_true():
    from experiments.config import ModelConfig
    mc = ModelConfig(name="x", load_in_8bit=True)
    assert mc.load_in_8bit is True


def _test_load_config_load_in_8bit_parses():
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: test_8bit
            model:
              name: meta-llama/Meta-Llama-3-70B
              dtype: bfloat16
              device: auto
              load_in_8bit: true
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              split:
                steering: 20
                validation: 20
                test: 60
            sweep:
              formulation: mcf
              num_shots: 5
            output:
              base_dir: /tmp/
        """)
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_config(str(cfg_path))
        assert cfg.model.load_in_8bit is True
        assert cfg.model.device == "auto"


def _test_load_config_load_in_8bit_absent_defaults_false():
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config_yaml(Path(tmp))
        cfg = load_config(cfg_path)
        assert cfg.model.load_in_8bit is False


# =============================================================================
# 9. SweepConfig: capture_batch_size field
# =============================================================================

def _test_sweep_config_capture_batch_size_default():
    from experiments.config import SweepConfig
    s = SweepConfig()
    assert s.capture_batch_size == 8, f"expected 8, got {s.capture_batch_size}"


def _test_load_config_capture_batch_size_parses():
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = _make_eval_csv(Path(tmp))
        neg_path  = _make_neg_csv(Path(tmp))
        yaml_text = textwrap.dedent(f"""
            experiment_id: test_capbatch
            model:
              name: meta-llama/Meta-Llama-3-8B
              dtype: float16
              device: cpu
            dataset:
              eval_path: {eval_path}
              neg_capture_path: {neg_path}
              split:
                steering: 20
                validation: 20
                test: 60
            sweep:
              formulation: mcf
              num_shots: 5
              capture_batch_size: 16
            output:
              base_dir: /tmp/
        """)
        cfg_path = Path(tmp) / "cfg.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_config(str(cfg_path))
        assert cfg.sweep.capture_batch_size == 16, f"expected 16, got {cfg.sweep.capture_batch_size}"


def _test_load_config_capture_batch_size_absent_defaults_8():
    from experiments.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _make_config_yaml(Path(tmp))
        cfg = load_config(cfg_path)
        assert cfg.sweep.capture_batch_size == 8


# =============================================================================
# 10. Batched capture in DifferenceInMeansSteering (CPU, fake model)
# =============================================================================

class _FakeModule(torch.nn.Module):
    """Transformer-like module: passes dummy hidden states through self.linear to trigger its hook."""
    def __init__(self, hidden=16):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)
        self.hidden = hidden

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        # Use input_ids to produce non-zero, input-dependent hidden states
        hidden_states = input_ids.float().unsqueeze(-1).expand(B, S, self.hidden) * 0.01
        return self.linear(hidden_states)  # actually calls linear → hook fires


def _make_dim_steerer(batch_size=4):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering
    from transformers import AutoTokenizer

    hidden = 16
    model = _FakeModule(hidden=hidden)
    mwh = ModelWithHooks(model)

    # Minimal tokenizer-like object backed by a real fast tokenizer
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    steerer = DifferenceInMeansSteering(
        model_with_hooks=mwh,
        tokenizer=tokenizer,
        target_layer="linear",
        token_position="mean",
        capture_batch_size=batch_size,
    )
    return steerer, hidden


def _test_batched_capture_produces_correct_count():
    steerer, _ = _make_dim_steerer(batch_size=3)
    prompts = [f"prompt number {i}" for i in range(10)]
    steerer.capture_positive_activations(prompts, max_length=32)
    assert len(steerer.positive_activations) == 10, \
        f"expected 10 vectors, got {len(steerer.positive_activations)}"


def _test_batched_capture_vector_shape():
    steerer, hidden = _make_dim_steerer(batch_size=4)
    prompts = [f"hello world {i}" for i in range(8)]
    steerer.capture_positive_activations(prompts, max_length=32)
    for i, vec in enumerate(steerer.positive_activations):
        assert vec.shape == (hidden,), f"vec {i} shape {vec.shape}, expected ({hidden},)"


def _test_batched_capture_batch_size_1_matches_batch_size_n():
    """Capture with batch_size=1 and batch_size=8 should produce identical vectors."""
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering
    from transformers import GPT2TokenizerFast

    hidden = 16
    torch.manual_seed(0)
    model = _FakeModule(hidden=hidden)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [f"test prompt {i}" for i in range(6)]

    def _run(bs):
        mwh = ModelWithHooks(model)
        s = DifferenceInMeansSteering(
            model_with_hooks=mwh, tokenizer=tokenizer,
            target_layer="linear", token_position="last",
            capture_batch_size=bs,
        )
        s.capture_positive_activations(prompts, max_length=32)
        return torch.stack(s.positive_activations)

    vecs_1 = _run(1)
    vecs_n = _run(8)
    assert torch.allclose(vecs_1, vecs_n, atol=1e-5), \
        f"batch_size=1 and batch_size=8 gave different activations"


def _test_batched_capture_uneven_last_batch():
    """10 prompts with batch_size=3 → batches of 3,3,3,1. All 10 should be captured."""
    steerer, _ = _make_dim_steerer(batch_size=3)
    prompts = [f"sentence {i}" for i in range(10)]
    steerer.capture_positive_activations(prompts, max_length=32)
    assert len(steerer.positive_activations) == 10


def _test_batched_capture_steering_vector_computable():
    steerer, hidden = _make_dim_steerer(batch_size=4)
    pos = [f"positive example {i}" for i in range(8)]
    neg = [f"negative example {i}" for i in range(8)]
    steerer.capture_positive_activations(pos, max_length=32)
    steerer.capture_negative_activations(neg, max_length=32)
    vec = steerer.compute_steering_vector(normalize=True, norm_type="unit")
    assert vec.shape == (hidden,)
    assert abs(vec.norm().item() - 1.0) < 1e-5, "unit-normalized vector should have norm ≈ 1"


# =============================================================================
# Run all
# =============================================================================

TESTS = [
    # SplitConfig
    ("SplitConfig: defaults",                           _test_split_config_defaults),
    ("SplitConfig: custom values",                      _test_split_config_custom),
    # split_dataset
    ("split_dataset: correct sizes (20/20/60 of 100)",  _test_split_sizes_exact),
    ("split_dataset: covers all rows",                  _test_split_sizes_cover_all),
    ("split_dataset: no overlap between splits",        _test_split_no_overlap),
    ("split_dataset: same seed → same split",           _test_split_reproducibility),
    ("split_dataset: different seeds → different split",_test_split_different_seeds),
    ("split_dataset: DataFrame values match source",    _test_split_dataframe_values_match),
    ("split_dataset: handles odd-size datasets",        _test_split_rounding_on_odd_n),
    # Config loading
    ("load_config: parses split section",               _test_load_config_with_split),
    ("load_config: reads sweep.seed",                   _test_load_config_seed_field),
    ("load_config: seed defaults to 42",                _test_load_config_seed_default),
    ("load_config: split=None when absent",             _test_load_config_without_split),
    # Validation
    ("validation: split sum != 100 → error",            _test_validation_split_sum_not_100),
    ("validation: split value = 0 → error",             _test_validation_split_zero_value),
    ("validation: split without neg_capture_path → error", _test_validation_split_requires_neg_capture_path),
    # Manifest
    ("manifest: eval_question_ids are contiguous",      _test_manifest_eval_ids_are_contiguous),
    ("manifest: original indices cover full dataset",   _test_manifest_original_indices_no_overlap),
    ("eval_df: split column has correct values",        _test_eval_df_split_column),
    # Neg capture
    ("neg capture: excludes few-shot rows",             _test_neg_capture_excludes_fewshot_rows),
    ("neg capture: reproducible with same seed",        _test_neg_capture_reproducible),
    ("neg capture: different seeds differ",             _test_neg_capture_different_seed),
    # W&B logging data pipeline
    ("wandb: compute_split_summary acc and delta",      _test_compute_split_summary_acc_and_delta),
    ("wandb: compute_split_summary n count",            _test_compute_split_summary_n),
    ("wandb: select_best_coef val_acc criterion",       _test_select_best_coef_val_acc),
    ("wandb: select_best_coef val_delta criterion",     _test_select_best_coef_val_delta),
    ("wandb: select_best_coef excludes coef=0",         _test_select_best_coef_excludes_baseline),
    ("wandb: select_best_coef tiebreak on delta_mean",  _test_select_best_coef_tiebreak_on_delta),
    ("wandb: select_best_coef independent per layer",   _test_select_best_coef_independent_per_layer),
    ("wandb: load_mcf_results attaches split labels",   _test_load_mcf_results_attaches_split),
    ("wandb: load_mcf_results no split col → no col",   _test_load_mcf_results_no_split_col),
    ("wandb: load_mcf_results no files → None",         _test_load_mcf_results_no_files_returns_none),
    ("wandb: load_cf_results attaches split labels",    _test_load_cf_results_attaches_split),
    ("wandb: load_cf_results no files → None",          _test_load_cf_results_no_files_returns_none),
    ("wandb: log_final_summary run=None is no-op",      _test_log_final_summary_run_none_is_noop),
    # ModelConfig: load_in_8bit
    ("ModelConfig: load_in_8bit defaults False",         _test_model_config_load_in_8bit_default),
    ("ModelConfig: load_in_8bit=True accepted",          _test_model_config_load_in_8bit_true),
    ("load_config: load_in_8bit parses from YAML",       _test_load_config_load_in_8bit_parses),
    ("load_config: load_in_8bit absent → False",         _test_load_config_load_in_8bit_absent_defaults_false),
    # SweepConfig: capture_batch_size
    ("SweepConfig: capture_batch_size defaults 8",       _test_sweep_config_capture_batch_size_default),
    ("load_config: capture_batch_size parses from YAML", _test_load_config_capture_batch_size_parses),
    ("load_config: capture_batch_size absent → 8",       _test_load_config_capture_batch_size_absent_defaults_8),
    # Batched capture
    ("batched capture: correct vector count",            _test_batched_capture_produces_correct_count),
    ("batched capture: vector shape matches hidden dim", _test_batched_capture_vector_shape),
    ("batched capture: batch=1 matches batch=N",         _test_batched_capture_batch_size_1_matches_batch_size_n),
    ("batched capture: uneven last batch handled",       _test_batched_capture_uneven_last_batch),
    ("batched capture: steering vector computable",      _test_batched_capture_steering_vector_computable),
]

if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} tests...\n")
    for name, fn in TESTS:
        test(name, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n{'='*60}")
    print(f"  {passed}/{len(_results)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n  --- {name} ---")
                print(tb)
    else:
        print("  — all green")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)
