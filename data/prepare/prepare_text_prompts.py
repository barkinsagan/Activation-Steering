"""
prepare_text_prompts.py — Clean and format raw textbook text into prompt TXT files.

Paste raw chapter text (or the output from the Claude extraction prompt) into
a plain .txt file, then run this script to filter, deduplicate, and write the
final prompt file used by geometry_sweep.py.

Output:
    data/prompts/physics_pos.txt
    data/prompts/biology_neg.txt

Usage:
    # From raw chapter text (auto sentence-split):
    python data/prepare/prepare_text_prompts.py \\
        --physics  raw/physics_chapter.txt \\
        --biology  raw/biology_chapter.txt

    # From Claude extraction output (already one sentence per line):
    python data/prepare/prepare_text_prompts.py \\
        --physics  raw/physics_extracted.txt \\
        --biology  raw/biology_extracted.txt \\
        --presplit

    # Append to existing files instead of overwriting:
    python data/prepare/prepare_text_prompts.py \\
        --physics  raw/physics_chapter2.txt \\
        --append
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sentence splitting and filtering
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on '. ', '! ', '? ' followed by a capital."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove reference markers like [1], (p. 42), Fig. 3, etc.
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(p\.?\s*\d+\)", "", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def load_lines(path: str) -> List[str]:
    """Load a file as pre-split lines (one sentence per line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def filter_sentences(
    sentences: List[str],
    min_words: int = 12,
    max_words: int = 70,
) -> List[str]:
    """
    Keep sentences that:
    - Fall within the word-count window
    - Are not mostly numeric / formulaic
    - Do not start with a lowercase letter (likely mid-sentence fragments)
    - Are not chapter headings (all caps or very short)
    """
    seen, out = set(), []
    for s in sentences:
        words = s.split()
        n = len(words)

        if not (min_words <= n <= max_words):
            continue
        if not s[0].isupper():
            continue
        # Skip if >25% of tokens are purely numeric/symbolic
        num_ratio = sum(1 for w in words if re.fullmatch(r"[\d\.\-\+\/\=\%\^\*]+", w)) / n
        if num_ratio > 0.25:
            continue
        # Deduplicate
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(
    input_path: Optional[str],
    output_path: Path,
    presplit: bool,
    append: bool,
    min_words: int,
    max_words: int,
    label: str,
) -> None:
    if input_path is None:
        return

    print(f"\nProcessing {label}: {input_path}")
    raw = Path(input_path).read_text(encoding="utf-8")

    if presplit:
        sentences = load_lines(input_path)
    else:
        sentences = split_sentences(raw)

    print(f"  Raw sentences:      {len(sentences)}")
    filtered = filter_sentences(sentences, min_words, max_words)
    print(f"  After filtering:    {len(filtered)}")

    if append and output_path.exists():
        existing = [l.strip() for l in output_path.read_text().splitlines() if l.strip()]
        combined = existing + filtered
        # Deduplicate across old + new
        seen, out = set(), []
        for s in combined:
            key = re.sub(r"\s+", " ", s.lower())
            if key not in seen:
                seen.add(key)
                out.append(s)
        filtered = out
        print(f"  After merge+dedup:  {len(filtered)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(filtered), encoding="utf-8")
    print(f"  Saved {len(filtered)} sentences → {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--physics",   default=None,
                    help="Input TXT file with physics text or extracted sentences")
    ap.add_argument("--biology",   default=None,
                    help="Input TXT file with biology text or extracted sentences")
    ap.add_argument("--presplit",  action="store_true",
                    help="Input is already one sentence per line (Claude extraction output)")
    ap.add_argument("--append",    action="store_true",
                    help="Append to existing output files rather than overwriting")
    ap.add_argument("--min_words", type=int, default=12)
    ap.add_argument("--max_words", type=int, default=70)
    ap.add_argument("--phys_out",  default="data/prompts/physics_pos.txt")
    ap.add_argument("--bio_out",   default="data/prompts/biology_neg.txt")
    args = ap.parse_args()

    if not args.physics and not args.biology:
        ap.error("Provide at least one of --physics or --biology")

    process(args.physics, Path(args.phys_out), args.presplit, args.append,
            args.min_words, args.max_words, "physics")
    process(args.biology, Path(args.bio_out),  args.presplit, args.append,
            args.min_words, args.max_words, "biology")


if __name__ == "__main__":
    main()
