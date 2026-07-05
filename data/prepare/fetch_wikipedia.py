"""
fetch_wikipedia.py — Download Wikipedia intro sections for physics and biology articles.

Uses the Wikipedia REST API (no auth, no rate limits for reasonable use).
Fetches the introductory text of each article, splits into sentences, filters
by length, and writes one sentence per line to a TXT file.

Output:
    data/eval/wikipedia_physics.txt
    data/eval/wikipedia_biology.txt

Usage:
    python data/prepare/fetch_wikipedia.py
    python data/prepare/fetch_wikipedia.py --n 120 --out_dir data/eval/
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List

import requests

# ---------------------------------------------------------------------------
# Article lists
# ---------------------------------------------------------------------------

PHYSICS_ARTICLES = [
    "Quantum_mechanics",
    "Special_relativity",
    "General_relativity",
    "Thermodynamics",
    "Electromagnetism",
    "Classical_mechanics",
    "Quantum_field_theory",
    "Statistical_mechanics",
    "Standard_Model",
    "Black_hole",
    "Higgs_boson",
    "Quantum_entanglement",
    "Wave%E2%80%93particle_duality",
    "Uncertainty_principle",
    "Schrödinger_equation",
    "Maxwell%27s_equations",
    "Conservation_of_energy",
    "Nuclear_physics",
    "Condensed_matter_physics",
    "Plasma_(physics)",
    "Superconductivity",
    "Photoelectric_effect",
    "Newton%27s_laws_of_motion",
    "Entropy",
    "Dark_matter",
    "Dark_energy",
    "Gravitational_wave",
    "Quantum_chromodynamics",
    "Particle_physics",
    "String_theory",
]

BIOLOGY_ARTICLES = [
    "Cell_(biology)",
    "DNA",
    "Evolution",
    "Natural_selection",
    "Genetics",
    "Protein",
    "Metabolism",
    "Photosynthesis",
    "Cellular_respiration",
    "Mitosis",
    "Meiosis",
    "Gene_expression",
    "Enzyme",
    "Immune_system",
    "Nervous_system",
    "Neuron",
    "Ecology",
    "Population_genetics",
    "Molecular_biology",
    "Biochemistry",
    "Eukaryote",
    "Prokaryote",
    "Virus",
    "Bacteria",
    "Mutation",
    "Chromosome",
    "RNA",
    "Ribosome",
    "Mitochondrion",
    "Cell_membrane",
]

# ---------------------------------------------------------------------------
# Wikipedia API
# ---------------------------------------------------------------------------

_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_HEADERS = {"User-Agent": "steering-vector-research/1.0 (academic; contact: research)"}


def fetch_summary(title: str) -> str:
    """Fetch the extract (intro text) for a Wikipedia article."""
    url = _API_URL.format(title=title)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("extract", "")
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch '{title}': {e}")
        return ""


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on '. ', '! ', '? '."""
    # Remove parenthetical citations like (1920–1958) and [1]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence-ending punctuation followed by space + capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def filter_sentences(sentences: List[str], min_words: int = 12, max_words: int = 70) -> List[str]:
    """Keep sentences in a word-count window and skip those with too many numbers."""
    out = []
    for s in sentences:
        words = s.split()
        if min_words <= len(words) <= max_words:
            # Skip sentences that are mostly numeric/formulaic
            num_ratio = sum(1 for w in words if re.match(r"^[\d\.\-\+\/\=]+$", w)) / len(words)
            if num_ratio < 0.25:
                out.append(s)
    return out


def fetch_sentences(articles: List[str], target_n: int) -> List[str]:
    """Fetch and filter sentences until we have target_n or exhaust the article list."""
    collected: List[str] = []
    for title in articles:
        if len(collected) >= target_n:
            break
        print(f"  Fetching: {title}")
        text = fetch_summary(title)
        if not text:
            continue
        sents = filter_sentences(split_sentences(text))
        collected.extend(sents)
        time.sleep(0.3)  # polite delay
    return collected[:target_n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",       type=int, default=120,
                    help="Target sentences per domain (default 120)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/eval"),
                    help="Output directory (default data/eval/)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for label, articles, out_name in [
        ("physics",  PHYSICS_ARTICLES,  "wikipedia_physics.txt"),
        ("biology",  BIOLOGY_ARTICLES,  "wikipedia_biology.txt"),
    ]:
        print(f"\nFetching {label} articles (target {args.n} sentences)...")
        sentences = fetch_sentences(articles, args.n)
        out_path = args.out_dir / out_name
        out_path.write_text("\n".join(sentences))
        print(f"  Saved {len(sentences)} sentences → {out_path}")


if __name__ == "__main__":
    main()
