"""
Preview MCF and CF prompt formatting for a given config.

Usage:
    python scripts/preview_format.py configs/exp_20260413_anatomy_llama8b_pilot.yaml
    python scripts/preview_format.py configs/exp_20260413_anatomy_llama8b_pilot.yaml --n 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import load_config
from experiments.registry import load_eval_dataset
from olmes.formatter import build_formatter


def main():
    parser = argparse.ArgumentParser(description="Preview prompt formatting for an experiment config.")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--n", type=int, default=2, help="Number of questions to preview (default: 2)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s = cfg.sweep

    df, _ = load_eval_dataset(cfg)
    formatter = build_formatter(
        task_prefix=s.task_prefix,
        num_shots=s.num_shots,
        fewshot_source=s.fewshot_source,
        shuffle_choices=s.shuffle_choices,
        seed=42,
    )

    print(f"\nConfig : {args.config}")
    print(f"Dataset: {cfg.dataset.eval_path}  ({len(df)} rows)")
    print(f"Shots  : {s.num_shots}")
    print(f"N      : {args.n}\n")

    print("=" * 70)
    print("MCF FORMAT")
    print("=" * 70)
    for i in range(args.n):
        row = df.iloc[i]
        result = formatter.format_mcf(row, question_idx=i)
        print(result.prompt)
        print(f"\n>>> correct label: {result.correct_label}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("CF FORMAT")
    print("=" * 70)
    for i in range(args.n):
        row = df.iloc[i]
        result = formatter.format_cf(row)
        print(result.prompt)
        print(f"\n>>> target : [{result.target}]")
        print(f">>> false1 : [{result.false1}]")
        print("-" * 70)


if __name__ == "__main__":
    main()
