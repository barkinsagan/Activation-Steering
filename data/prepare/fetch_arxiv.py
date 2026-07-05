"""
fetch_arxiv.py — Download arXiv abstracts for physics and biology (q-bio).

Uses the arXiv API (no auth required). Fetches recent abstracts from physics
and quantitative biology categories, writes one abstract per line to TXT files.

Output:
    data/eval/arxiv_physics.txt
    data/eval/biorxiv_biology.txt   (q-bio categories from arXiv)

Usage:
    python data/prepare/fetch_arxiv.py
    python data/prepare/fetch_arxiv.py --n 120 --out_dir data/eval/
"""

from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import requests

# ---------------------------------------------------------------------------
# Category lists
# ---------------------------------------------------------------------------

PHYSICS_CATEGORIES = [
    "cond-mat.str-el",    # strongly correlated electrons
    "hep-ph",             # high energy physics – phenomenology
    "quant-ph",           # quantum physics
    "physics.gen-ph",     # general physics
    "gr-qc",              # general relativity and quantum cosmology
    "cond-mat.mes-hall",  # mesoscale and nanoscale physics
    "physics.optics",     # optics
    "nucl-th",            # nuclear theory
]

BIOLOGY_CATEGORIES = [
    "q-bio.BM",   # biomolecules
    "q-bio.CB",   # cell behavior
    "q-bio.GN",   # genomics
    "q-bio.PE",   # populations and evolution
    "q-bio.MN",   # molecular networks
    "q-bio.NC",   # neurons and cognition
]

_API_URL = "https://export.arxiv.org/api/query"
_NS      = {"atom": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace and remove newlines inside abstracts."""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def fetch_abstracts(categories: List[str], target_n: int) -> List[str]:
    collected: List[str] = []
    per_cat = max(1, -(-target_n // len(categories)))  # ceil division

    for cat in categories:
        if len(collected) >= target_n:
            break
        print(f"  Fetching category: {cat}  (requesting {per_cat})")
        params = {
            "search_query": f"cat:{cat}",
            "start":        0,
            "max_results":  per_cat,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }
        try:
            r = requests.get(_API_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] {cat}: {e}")
            time.sleep(2)
            continue

        root = ET.fromstring(r.text)
        entries = root.findall("atom:entry", _NS)
        for entry in entries:
            summary_el = entry.find("atom:summary", _NS)
            if summary_el is None or not summary_el.text:
                continue
            abstract = _clean(summary_el.text)
            # Basic quality filter: skip very short or very long abstracts
            words = abstract.split()
            if 60 <= len(words) <= 350:
                collected.append(abstract)

        print(f"    → {len(entries)} entries fetched, {len(collected)} total so far")
        time.sleep(3)  # arXiv asks for 3s between requests

    return collected[:target_n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",       type=int, default=120,
                    help="Target abstracts per domain (default 120)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/eval"),
                    help="Output directory (default data/eval/)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for label, categories, out_name in [
        ("physics", PHYSICS_CATEGORIES, "arxiv_physics.txt"),
        ("biology", BIOLOGY_CATEGORIES, "biorxiv_biology.txt"),
    ]:
        print(f"\nFetching {label} abstracts (target {args.n})...")
        abstracts = fetch_abstracts(categories, args.n)
        out_path  = args.out_dir / out_name
        out_path.write_text("\n".join(abstracts))
        print(f"  Saved {len(abstracts)} abstracts → {out_path}")


if __name__ == "__main__":
    main()
