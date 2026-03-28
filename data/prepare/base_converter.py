"""
Shared utilities for dataset converters.
All converters output a standard CSV with columns: prompt, target, false1, false2, false3
(false columns optional — omit for open-ended tasks).
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Optional


STANDARD_COLUMNS_MCQ = ["prompt", "target", "false1", "false2", "false3"]
STANDARD_COLUMNS_CZ  = ["prompt", "target", "false1", "false2", "false3"]  # same structure


def shuffle_and_sample(data: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """Reproducibly shuffle and take up to n items."""
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    return data[:n]


def save_csv(rows: List[Dict], output_path: Path, columns: List[str]) -> None:
    """Write rows to a CSV file, creating parent dirs as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows → {output_path}")


def validate_rows(rows: List[Dict], require_false: bool = True) -> List[Dict]:
    """Drop rows missing required fields and log how many were dropped."""
    required = ["prompt", "target"]
    if require_false:
        required += ["false1", "false2", "false3"]
    clean = [r for r in rows if all(r.get(c) for c in required)]
    dropped = len(rows) - len(clean)
    if dropped:
        print(f"  Warning: dropped {dropped} rows with missing fields")
    return clean
