"""
Shared utilities for dataset converters.
All converters output a standard CSV with columns: prompt, target, false1, false2, false3
(false columns optional — omit for open-ended tasks).

Few-shot YAML format:
    - prompt: "Which bone is the longest in the human body?"
      choices:
        - "Femur"      # A
        - "Tibia"      # B
        - "Humerus"    # C
        - "Fibula"     # D
      correct: 0       # 0-based index of the correct choice
"""

import csv
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional


STANDARD_COLUMNS_MCQ = ["prompt", "target", "false1", "false2", "false3"]
STANDARD_COLUMNS_CZ  = ["prompt", "target", "false1", "false2", "false3"]  # same structure

# Cyclic label assignment for few-shot examples: ensures label balance across 5 shots
# Row 0→A, 1→B, 2→C, 3→D, 4→A
FEWSHOT_CORRECT_POSITIONS = [0, 1, 2, 3, 0]


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


def make_fewshot_examples(rows: List[Dict], n: int = 5) -> List[Dict]:
    """
    Convert the first n rows (from train split, pre-shuffle) into OLMES-style
    few-shot YAML examples with balanced label assignment.

    Correct answer is placed at position FEWSHOT_CORRECT_POSITIONS[i] (0-based),
    which cycles A→B→C→D→A for 5 shots, guaranteeing label balance.

    Args:
        rows: Standard MCQ rows with prompt, target, false1, false2, false3.
        n:    Number of few-shot examples to generate (default 5).

    Returns:
        List of dicts ready for YAML serialisation.
    """
    examples = []
    for idx, row in enumerate(rows[:n]):
        correct_pos = FEWSHOT_CORRECT_POSITIONS[idx % len(FEWSHOT_CORRECT_POSITIONS)]
        falses = [row["false1"], row["false2"], row["false3"]]
        # Build choices list: insert target at correct_pos, fill rest with falses
        choices: List[Optional[str]] = [None] * 4
        choices[correct_pos] = row["target"]
        false_iter = iter(falses)
        for j in range(4):
            if choices[j] is None:
                choices[j] = next(false_iter)
        examples.append({
            "prompt":  row["prompt"],
            "choices": choices,
            "correct": correct_pos,
        })
    return examples


def save_yaml(examples: List[Dict], output_path: Path) -> None:
    """Write few-shot examples to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(examples, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  Saved {len(examples)} few-shot examples → {output_path}")
