"""
MedMCQA → standard eval CSV converter.

HuggingFace dataset: openlifescienceai/medmcqa
Fields used: question, opa, opb, opc, opd, cop (0=A, 1=B, 2=C, 3=D), subject_name

Outputs (per subject and for the mixed "all" set):
  data/eval/{subject}_sweep_150.csv   — 150 samples for layer/coef sweeps
  data/eval/{subject}_eval_500.csv    — 500 samples for final evaluation

MCQ and CZ share the same CSV structure:
  prompt, target, false1, false2, false3
The formulation field in the experiment YAML controls MCF vs CF scoring.
Few-shot examples are generated into data/fewshots/ per subject.

Sweep CSVs come from the train split (hyperparameter search: layers, coefs).
Eval CSVs come from the validation split (final reporting).
Note: MedMCQA test split has no public labels, so validation is used as held-out eval.

Usage:
  # Specific subjects
  python data/prepare/convert_medmcqa.py --subjects Anatomy Pharmacology

  # All subjects + mixed set
  python data/prepare/convert_medmcqa.py --all

  # Custom sizes or splits
  python data/prepare/convert_medmcqa.py --all --sweep_n 100 --eval_n 300
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.prepare.base_converter import (
    shuffle_and_sample,
    save_csv,
    save_yaml,
    validate_rows,
    make_fewshot_examples,
    STANDARD_COLUMNS_MCQ,
)

# Canonical subject name → safe filename slug
SUBJECT_SLUGS = {
    "Anatomy": "anatomy",
    "Physiology": "physiology",
    "Biochemistry": "biochemistry",
    "Pathology": "pathology",
    "Pharmacology": "pharmacology",
    "Microbiology": "microbiology",
    "Forensic Medicine": "forensic_medicine",
    "Medicine": "medicine",
    "Surgery": "surgery",
    "Obstetrics & Gynaecology": "obs_gynae",
    "Paediatrics": "paediatrics",
    "Radiology": "radiology",
    "Ophthalmology": "ophthalmology",
    "ENT": "ent",
    "Anaesthesia": "anaesthesia",
    "Psychiatry": "psychiatry",
    "Dermatology": "dermatology",
    "Orthopaedics": "orthopaedics",
    "Community Medicine": "community_medicine",
    "Dental": "dental",
}

OPTION_KEYS = ["opa", "opb", "opc", "opd"]


def row_to_standard(item: Dict) -> Optional[Dict]:
    """Convert one MedMCQA item to standard CSV row. Returns None if malformed."""
    try:
        cop = int(item["cop"])  # 0-3
        options = [item[k] for k in OPTION_KEYS]
        if any(o is None or str(o).strip() == "" for o in options):
            return None
        target = options[cop]
        falses = [options[i] for i in range(4) if i != cop]
        return {
            "prompt": str(item["question"]).strip(),
            "target": str(target).strip(),
            "false1": str(falses[0]).strip(),
            "false2": str(falses[1]).strip(),
            "false3": str(falses[2]).strip(),
        }
    except (KeyError, ValueError, TypeError):
        return None


def load_medmcqa(split: str):
    """Load a MedMCQA split from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading MedMCQA ({split} split) from HuggingFace...")
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    print(f"  Loaded {len(ds)} items")
    return ds


def convert_subjects(
    subjects: List[str],
    sweep_ds,
    eval_ds,
    sweep_n: int,
    eval_n: int,
    out_dir: Path,
    fewshot_dir: Path,
    fewshot_n: int = 5,
    seed: int = 42,
):
    """Convert specific subjects and save sweep + eval CSVs and few-shot YAMLs."""
    def collect(ds):
        rows: Dict[str, List[Dict]] = {s: [] for s in subjects}
        for item in ds:
            subj = item.get("subject_name", "")
            if subj in rows:
                row = row_to_standard(item)
                if row is not None:
                    rows[subj].append(row)
        return rows

    sweep_rows_by_subj = collect(sweep_ds)
    eval_rows_by_subj  = collect(eval_ds)

    for subj in subjects:
        slug = SUBJECT_SLUGS.get(subj, subj.lower().replace(" ", "_"))

        s_rows = validate_rows(sweep_rows_by_subj[subj])
        e_rows = validate_rows(eval_rows_by_subj[subj])
        print(f"\n[{subj}] train={len(s_rows)} validation={len(e_rows)}")

        # Few-shot examples: take first fewshot_n rows BEFORE the sweep shuffle
        # so they are guaranteed not to appear in the sweep CSV
        if fewshot_n > 0 and len(s_rows) >= fewshot_n:
            fewshot_rows = s_rows[:fewshot_n]
            examples = make_fewshot_examples(fewshot_rows, n=fewshot_n)
            save_yaml(examples, fewshot_dir / f"{slug}_fewshot.yaml")
            # Exclude fewshot rows from the sweep pool
            s_rows_for_sweep = s_rows[fewshot_n:]
        else:
            s_rows_for_sweep = s_rows

        save_csv(shuffle_and_sample(s_rows_for_sweep, sweep_n, seed=seed),
                 out_dir / f"{slug}_sweep_{sweep_n}.csv", STANDARD_COLUMNS_MCQ)
        save_csv(shuffle_and_sample(e_rows, eval_n, seed=seed),
                 out_dir / f"{slug}_eval_{eval_n}.csv", STANDARD_COLUMNS_MCQ)


def convert_all(sweep_ds, eval_ds, sweep_n: int, eval_n: int,
               out_dir: Path, fewshot_dir: Path, fewshot_n: int = 5, seed: int = 42):
    """Convert all subjects combined into a mixed 'medmcqa_all' set."""
    def collect(ds):
        rows = []
        for item in ds:
            row = row_to_standard(item)
            if row is not None:
                rows.append(row)
        return rows

    s_rows = validate_rows(collect(sweep_ds))
    e_rows = validate_rows(collect(eval_ds))
    print(f"\n[All subjects combined] train={len(s_rows)} validation={len(e_rows)}")

    if fewshot_n > 0 and len(s_rows) >= fewshot_n:
        examples = make_fewshot_examples(s_rows[:fewshot_n], n=fewshot_n)
        save_yaml(examples, fewshot_dir / "medmcqa_all_fewshot.yaml")
        s_rows_for_sweep = s_rows[fewshot_n:]
    else:
        s_rows_for_sweep = s_rows

    save_csv(shuffle_and_sample(s_rows_for_sweep, sweep_n, seed=seed),
             out_dir / f"medmcqa_all_sweep_{sweep_n}.csv", STANDARD_COLUMNS_MCQ)
    save_csv(shuffle_and_sample(e_rows, eval_n, seed=seed),
             out_dir / f"medmcqa_all_eval_{eval_n}.csv", STANDARD_COLUMNS_MCQ)


def _resolve_subjects(names: List[str]) -> List[str]:
    """Resolve subject name strings to canonical keys, case-insensitive."""
    slug_to_canonical = {v: k for k, v in SUBJECT_SLUGS.items()}
    slug_to_canonical.update({k.lower(): k for k in SUBJECT_SLUGS})
    resolved = []
    for s in names:
        canonical = s if s in SUBJECT_SLUGS else slug_to_canonical.get(s.lower())
        if not canonical:
            print(f"Warning: unknown subject '{s}'. Known: {list(SUBJECT_SLUGS.keys())}")
        else:
            resolved.append(canonical)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Convert MedMCQA to standard eval CSV")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject names to convert (e.g. Anatomy Pharmacology)")
    parser.add_argument("--all", action="store_true",
                        help="Convert all subjects + mixed set")
    parser.add_argument("--sweep_split", default="train",
                        help="HuggingFace split for sweep CSVs (default: train)")
    parser.add_argument("--eval_split", default="validation",
                        help="HuggingFace split for eval CSVs (default: validation)")
    parser.add_argument("--sweep_n", type=int, default=150,
                        help="Samples per subject for sweep CSV (default: 150)")
    parser.add_argument("--eval_n", type=int, default=500,
                        help="Samples per subject for eval CSV (default: 500)")
    parser.add_argument("--fewshot_n", type=int, default=5,
                        help="Few-shot examples per subject (default: 5; 0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data/eval",
                        help="Output directory for CSVs (default: data/eval)")
    parser.add_argument("--fewshot_dir", default="data/fewshots",
                        help="Output directory for few-shot YAMLs (default: data/fewshots)")
    args = parser.parse_args()

    if not args.subjects and not args.all:
        parser.error("Specify --subjects or --all")

    out_dir      = Path(args.out_dir)
    fewshot_dir  = Path(args.fewshot_dir)
    sweep_ds = load_medmcqa(args.sweep_split)
    eval_ds  = load_medmcqa(args.eval_split)

    if args.subjects:
        resolved = _resolve_subjects(args.subjects)
        if resolved:
            convert_subjects(resolved, sweep_ds, eval_ds,
                             args.sweep_n, args.eval_n, out_dir,
                             fewshot_dir, args.fewshot_n, args.seed)

    if args.all:
        convert_subjects(list(SUBJECT_SLUGS.keys()), sweep_ds, eval_ds,
                         args.sweep_n, args.eval_n, out_dir,
                         fewshot_dir, args.fewshot_n, args.seed)
        convert_all(sweep_ds, eval_ds, args.sweep_n, args.eval_n, out_dir,
                    fewshot_dir, args.fewshot_n, args.seed)


if __name__ == "__main__":
    main()
