"""
MMLU → standard eval CSV converter.

HuggingFace dataset: cais/mmlu
Fields used: question, choices (list of 4), answer (int 0-3)

Outputs per subject:
  data/eval/mmlu_{slug}_sweep.csv  — sweep CSV for layer/coef search
  data/fewshots/mmlu_{slug}_fewshot.yaml

Usage:
  # Non-medical subjects for DIM neg prompts
  python data/prepare/convert_mmlu.py --subjects world_history high_school_geography

  # All non-medical subjects combined into one file
  python data/prepare/convert_mmlu.py --subjects world_history high_school_geography \
      european_history human_geography prehistory --merge --out data/eval/mmlu_nonmed_sweep.csv

  # Custom size
  python data/prepare/convert_mmlu.py --subjects world_history --sweep_n 200
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.prepare.base_converter import (
    shuffle_and_sample,
    save_csv,
    save_yaml,
    validate_rows,
    make_fewshot_examples,
    STANDARD_COLUMNS_MCQ,
)

# Recommended non-medical subjects for use as neg contrast class
NON_MEDICAL_SUBJECTS = [
    "world_history",
    "high_school_geography",
    "european_history",
    "human_geography",
    "prehistory",
    "world_religions",
    "high_school_world_history",
    "sociology",
    "philosophy",
    "jurisprudence",
]


def load_mmlu(subject: str, split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, trust_remote_code=True)
    return ds[split]


def convert_row(row: dict) -> Dict:
    choices = row["choices"]
    answer_idx = int(row["answer"])
    correct = choices[answer_idx]
    false_choices = [c for i, c in enumerate(choices) if i != answer_idx]
    return {
        "prompt": row["question"].strip(),
        "target": correct.strip(),
        "false1": false_choices[0].strip(),
        "false2": false_choices[1].strip(),
        "false3": false_choices[2].strip(),
    }


def convert_subject(subject: str, sweep_n: int, fewshot_n: int,
                    out_dir: Path, fewshot_dir: Path, seed: int = 42):
    print(f"\nConverting: {subject}")
    slug = subject.replace(" ", "_").lower()

    try:
        ds = load_mmlu(subject, split="test")
    except Exception:
        print(f"  'test' split not found for {subject}, trying 'validation'")
        ds = load_mmlu(subject, split="validation")

    rows = [convert_row(r) for r in ds]
    rows = validate_rows(rows)
    print(f"  Loaded {len(rows)} rows")

    if fewshot_n > 0 and len(rows) > fewshot_n:
        examples = make_fewshot_examples(rows[:fewshot_n], n=fewshot_n)
        save_yaml(examples, fewshot_dir / f"mmlu_{slug}_fewshot.yaml")
        rows_for_sweep = rows[fewshot_n:]
    else:
        rows_for_sweep = rows

    sweep_rows = shuffle_and_sample(rows_for_sweep, sweep_n, seed=seed)
    save_csv(sweep_rows, out_dir / f"mmlu_{slug}_sweep.csv", STANDARD_COLUMNS_MCQ)
    return sweep_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=NON_MEDICAL_SUBJECTS,
                    help="MMLU subject names (default: all recommended non-medical)")
    ap.add_argument("--sweep_n", type=int, default=150)
    ap.add_argument("--fewshot_n", type=int, default=5)
    ap.add_argument("--merge", action="store_true",
                    help="Also write a combined CSV of all subjects")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path for merged CSV (implies --merge)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    ap.add_argument("--fewshot-dir", type=Path, default=Path("data/fewshots"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fewshot_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for subject in args.subjects:
        rows = convert_subject(
            subject, args.sweep_n, args.fewshot_n,
            args.out_dir, args.fewshot_dir, args.seed,
        )
        all_rows.extend(rows)

    if args.merge or args.out:
        out_path = args.out or args.out_dir / "mmlu_nonmed_sweep.csv"
        merged = shuffle_and_sample(all_rows, len(all_rows), seed=args.seed)
        save_csv(merged, out_path, STANDARD_COLUMNS_MCQ)
        print(f"\nMerged {len(merged)} rows → {out_path}")


if __name__ == "__main__":
    main()
