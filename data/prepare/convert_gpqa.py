"""
GPQA → standard eval CSV converter.

HuggingFace dataset: Idavidrein/gpqa
Subsets: gpqa_main (448 q), gpqa_diamond (198 q, hardest), gpqa_extended (546 q)
Fields used: Question, Correct Answer, Incorrect Answer 1/2/3, Domain, Subdomain

GPQA has only a single "train" split, so we perform our own sweep/eval split
via a reproducible shuffle (seed=42) with a configurable split ratio (default 70/30).

Outputs (per domain and for the combined set):
  data/eval/{slug}_sweep.csv   — sweep samples for layer/coef tuning
  data/eval/{slug}_eval.csv    — held-out samples for final evaluation
  data/fewshots/{slug}_fewshot.yaml

MCQ CSV structure:  prompt, target, false1, false2, false3

Domains: Biology, Chemistry, Physics

Usage:
  # Specific domains
  python data/prepare/convert_gpqa.py --domains Biology Chemistry

  # All domains + combined set
  python data/prepare/convert_gpqa.py --all

  # Different subset (default: gpqa_main)
  python data/prepare/convert_gpqa.py --all --subset gpqa_diamond

  # Custom sizes
  python data/prepare/convert_gpqa.py --all --sweep_n 80 --eval_n 100
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.prepare.base_converter import (
    shuffle_and_sample,
    save_csv,
    save_yaml,
    validate_rows,
    make_fewshot_examples,
    STANDARD_COLUMNS_MCQ,
)

VALID_SUBSETS = ["gpqa_main", "gpqa_diamond", "gpqa_extended"]

DOMAIN_SLUGS = {
    "Biology":   "biology",
    "Chemistry": "chemistry",
    "Physics":   "physics",
}


def row_to_standard(item: Dict) -> Optional[Dict]:
    """Convert one GPQA item to standard CSV row. Returns None if malformed."""
    try:
        question = str(item["Question"]).strip()
        target   = str(item["Correct Answer"]).strip()
        false1   = str(item["Incorrect Answer 1"]).strip()
        false2   = str(item["Incorrect Answer 2"]).strip()
        false3   = str(item["Incorrect Answer 3"]).strip()
        if not all([question, target, false1, false2, false3]):
            return None
        return {
            "prompt": question,
            "target": target,
            "false1": false1,
            "false2": false2,
            "false3": false3,
        }
    except (KeyError, TypeError):
        return None


def load_gpqa(subset: str):
    """Load a GPQA subset from HuggingFace datasets (single 'train' split)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    if subset not in VALID_SUBSETS:
        print(f"Error: unknown subset '{subset}'. Valid: {VALID_SUBSETS}")
        sys.exit(1)

    print(f"Loading GPQA subset='{subset}' from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", subset, split="train")
    print(f"  Loaded {len(ds)} items")
    return ds


def _sweep_eval_split(rows: List[Dict], sweep_n: int, eval_n: int, seed: int):
    """
    Split a single pool into non-overlapping sweep and eval sets.
    Sweep draws from the first 70% of the shuffled pool; eval from the remaining 30%.
    """
    import random
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    split = max(1, int(len(pool) * 0.7))
    sweep_pool = pool[:split]
    eval_pool  = pool[split:]
    return shuffle_and_sample(sweep_pool, sweep_n, seed=seed), \
           shuffle_and_sample(eval_pool,  eval_n,  seed=seed)


def convert_domains(
    domains: List[str],
    ds,
    sweep_n: int,
    eval_n: int,
    subset: str,
    out_dir: Path,
    fewshot_dir: Path,
    fewshot_n: int = 5,
    seed: int = 42,
):
    """Convert specific domains and save sweep + eval CSVs and few-shot YAMLs."""
    rows_by_domain: Dict[str, List[Dict]] = {d: [] for d in domains}

    for item in ds:
        domain = item.get("Domain", "")
        if domain in rows_by_domain:
            row = row_to_standard(item)
            if row is not None:
                rows_by_domain[domain].append(row)

    for domain in domains:
        slug = DOMAIN_SLUGS.get(domain, domain.lower())
        prefix = f"gpqa_{subset.replace('gpqa_', '')}_{slug}"

        all_rows = validate_rows(rows_by_domain[domain])
        print(f"\n[{domain}] total={len(all_rows)}")

        if fewshot_n > 0 and len(all_rows) >= fewshot_n:
            fewshot_rows = all_rows[:fewshot_n]
            examples = make_fewshot_examples(fewshot_rows, n=fewshot_n)
            save_yaml(examples, fewshot_dir / f"{prefix}_fewshot.yaml")
            pool = all_rows[fewshot_n:]
        else:
            pool = all_rows

        sweep_rows, eval_rows = _sweep_eval_split(pool, sweep_n, eval_n, seed)
        save_csv(sweep_rows, out_dir / f"{prefix}_sweep.csv", STANDARD_COLUMNS_MCQ)
        save_csv(eval_rows,  out_dir / f"{prefix}_eval.csv",  STANDARD_COLUMNS_MCQ)


def convert_general(
    ds,
    steering_n_per_domain: int,
    subset: str,
    out_dir: Path,
    fewshot_dir: Path,
    fewshot_n: int = 5,
    seed: int = 42,
):
    """
    Build a merged 'general science' CSV with a fixed structure:
      rows 0   .. steering_n*3-1  : steering rows (steering_n from each domain, in order Bio→Chem→Phys)
      rows steering_n*3 .. end    : eval rows (all remaining from every domain, shuffled together)

    The runner reads this as: capture_n = steering_n*3 rows for positive DIM capture,
    everything after for evaluation.

    fewshot_n rows are skipped from the top of each domain's shuffled pool so they
    don't bleed into the steering or eval sets.
    """
    import random

    DOMAINS = ["Biology", "Chemistry", "Physics"]
    rows_by_domain: Dict[str, List[Dict]] = {d: [] for d in DOMAINS}

    for item in ds:
        domain = item.get("Domain", "")
        if domain in rows_by_domain:
            row = row_to_standard(item)
            if row is not None:
                rows_by_domain[domain].append(row)

    steering_rows: List[Dict] = []
    eval_rows:     List[Dict] = []

    for domain in DOMAINS:
        rows = validate_rows(rows_by_domain[domain])
        rng = random.Random(seed)
        rng.shuffle(rows)
        # Skip fewshot_n rows for consistency with per-domain converter
        pool = rows[fewshot_n:] if fewshot_n > 0 and len(rows) > fewshot_n else rows
        steering_rows.extend(pool[:steering_n_per_domain])
        eval_rows.extend(pool[steering_n_per_domain:])
        actual_s = min(steering_n_per_domain, len(pool))
        actual_e = max(0, len(pool) - steering_n_per_domain)
        print(f"  [{domain}] steering={actual_s}  eval={actual_e}")

    # Shuffle eval rows so domains are interleaved
    random.Random(seed).shuffle(eval_rows)

    combined = steering_rows + eval_rows
    label  = subset.replace("gpqa_", "")
    prefix = f"gpqa_{label}_general"

    print(f"\n[General science] steering={len(steering_rows)} ({steering_n_per_domain}/domain)"
          f"  eval={len(eval_rows)}  total={len(combined)}")

    save_csv(combined, out_dir / f"{prefix}_sweep.csv", STANDARD_COLUMNS_MCQ)


def convert_all_combined(
    ds,
    sweep_n: int,
    eval_n: int,
    subset: str,
    out_dir: Path,
    fewshot_dir: Path,
    fewshot_n: int = 5,
    seed: int = 42,
):
    """Convert all domains combined into a single mixed set."""
    all_rows = []
    for item in ds:
        row = row_to_standard(item)
        if row is not None:
            all_rows.append(row)

    all_rows = validate_rows(all_rows)
    label = subset.replace("gpqa_", "")
    prefix = f"gpqa_{label}_all"
    print(f"\n[All domains combined] total={len(all_rows)}")

    if fewshot_n > 0 and len(all_rows) >= fewshot_n:
        examples = make_fewshot_examples(all_rows[:fewshot_n], n=fewshot_n)
        save_yaml(examples, fewshot_dir / f"{prefix}_fewshot.yaml")
        pool = all_rows[fewshot_n:]
    else:
        pool = all_rows

    sweep_rows, eval_rows = _sweep_eval_split(pool, sweep_n, eval_n, seed)
    save_csv(sweep_rows, out_dir / f"{prefix}_sweep.csv", STANDARD_COLUMNS_MCQ)
    save_csv(eval_rows,  out_dir / f"{prefix}_eval.csv",  STANDARD_COLUMNS_MCQ)


def main():
    parser = argparse.ArgumentParser(description="Convert GPQA to standard eval CSV")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domains to convert: Biology Chemistry Physics")
    parser.add_argument("--all", action="store_true",
                        help="Convert all domains + combined set")
    parser.add_argument("--general", action="store_true",
                        help="Build merged general-science CSV (steering_n per domain + remaining eval)")
    parser.add_argument("--steering_n", type=int, default=50,
                        help="Steering rows per domain for --general (default: 50)")
    parser.add_argument("--subset", default="gpqa_main",
                        choices=VALID_SUBSETS,
                        help="GPQA subset to use (default: gpqa_main)")
    parser.add_argument("--sweep_n", type=int, default=100,
                        help="Max samples for sweep CSV (default: 100)")
    parser.add_argument("--eval_n", type=int, default=100,
                        help="Max samples for eval CSV (default: 100)")
    parser.add_argument("--fewshot_n", type=int, default=5,
                        help="Few-shot examples per domain (default: 5; 0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data/eval")
    parser.add_argument("--fewshot_dir", default="data/fewshots")
    args = parser.parse_args()

    if not args.domains and not args.all and not args.general:
        parser.error("Specify --domains, --all, or --general")

    out_dir     = Path(args.out_dir)
    fewshot_dir = Path(args.fewshot_dir)
    ds = load_gpqa(args.subset)

    if args.domains:
        known = set(DOMAIN_SLUGS.keys())
        resolved = []
        for d in args.domains:
            match = d if d in known else next((k for k in known if k.lower() == d.lower()), None)
            if match:
                resolved.append(match)
            else:
                print(f"Warning: unknown domain '{d}'. Known: {list(known)}")
        if resolved:
            convert_domains(resolved, ds, args.sweep_n, args.eval_n, args.subset,
                            out_dir, fewshot_dir, args.fewshot_n, args.seed)

    if args.all:
        convert_domains(list(DOMAIN_SLUGS.keys()), ds, args.sweep_n, args.eval_n,
                        args.subset, out_dir, fewshot_dir, args.fewshot_n, args.seed)
        convert_all_combined(ds, args.sweep_n, args.eval_n, args.subset,
                             out_dir, fewshot_dir, args.fewshot_n, args.seed)

    if args.general:
        convert_general(ds, args.steering_n, args.subset,
                        out_dir, fewshot_dir, args.fewshot_n, args.seed)


if __name__ == "__main__":
    main()
