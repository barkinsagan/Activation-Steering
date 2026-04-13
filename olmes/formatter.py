"""
OLMES-compliant prompt formatter.

Two formulations per task:
  MCF (Multiple Choice Formulation):
      Full labeled prompt with all choices injected; score the single label token.
  CF  (Cloze/Completion Formulation):
      Bare question + Answer: prefix; score each answer continuation separately.

Reference: https://github.com/allenai/olmes  (Apache 2.0)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


LABELS: List[str] = ["A", "B", "C", "D"]

# Dataset-specific question prefixes.
# None means no prefix (HellaSwag/WinoGrande CF — pure language modelling).
TASK_PREFIXES: Dict[str, Optional[str]] = {
    "question":          "Question",        # default for most MCQA
    "goal":              "Goal",            # PIQA
    "fill_in_the_blank": "Fill in the blank",  # WinoGrande MCF
    "continuation":      None,              # HellaSwag / WinoGrande CF
}


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class MCFFormattedRow:
    """Output of OLMESFormatter.format_mcf()."""
    prompt: str                     # Full prompt ending with "Answer:"
    correct_label: str              # "A", "B", "C", or "D"
    label_to_choice: Dict[str, str] # {"A": "iron", "B": "basalt", ...}


@dataclass
class CFFormattedRow:
    """Output of OLMESFormatter.format_cf()."""
    prompt: str   # "Question: ...\nAnswer: " (trailing space)
    target: str   # " iron"  (leading space for SentencePiece tokenisers)
    false1: str   # " basalt"
    false2: str   # " magma"
    false3: str   # " quartz"


# =============================================================================
# Formatter
# =============================================================================

class OLMESFormatter:
    """
    OLMES-compliant formatter for MCF and CF evaluation.

    MCF prompt example (5-shot omitted for brevity):
        Question: Earth's core is primarily composed of which material?
         A. basalt
         B. iron
         C. magma
         D. quartz
        Answer:

    CF prompt example:
        Question: Earth's core is primarily composed of which material?
        Answer: <candidate appended per candidate>

    Few-shot examples are prepended and separated by double newlines (\\n\\n).
    """

    def __init__(
        self,
        task_prefix: str = "question",
        num_shots: int = 5,
        fewshot_examples: Optional[List[Dict]] = None,
        shuffle_choices: bool = True,
        seed: int = 42,
    ):
        if task_prefix not in TASK_PREFIXES:
            raise ValueError(
                f"Unknown task_prefix={task_prefix!r}. "
                f"Valid options: {list(TASK_PREFIXES)}"
            )
        self.task_prefix = task_prefix
        self.num_shots = num_shots
        self.fewshot_examples = fewshot_examples or []
        self.shuffle_choices = shuffle_choices
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal prompt builders
    # ------------------------------------------------------------------

    def _prefix_str(self) -> Optional[str]:
        return TASK_PREFIXES[self.task_prefix]

    def _build_mcf_body(
        self,
        question: str,
        choices: List[str],
        answer_label: Optional[str] = None,
    ) -> str:
        """
        Build one MCF block.

        answer_label: if provided, appended after "Answer:" (used for few-shot shots).
        """
        prefix = self._prefix_str()
        lines: List[str] = []

        if prefix:
            lines.append(f"{prefix}: {question.strip()}")
        else:
            lines.append(question.strip())

        for label, choice in zip(LABELS, choices):
            lines.append(f" {label}. {choice.strip()}")

        if answer_label is not None:
            lines.append(f"Answer: {answer_label}")
        else:
            lines.append("Answer:")

        return "\n".join(lines)

    def _build_cf_body(
        self,
        question: str,
        answer_text: Optional[str] = None,
    ) -> str:
        """
        Build one CF block.

        answer_text: if provided, appended after "Answer: " (used for few-shot shots).
        """
        prefix = self._prefix_str()
        q_line = f"{prefix}: {question.strip()}" if prefix else question.strip()

        if answer_text is not None:
            return f"{q_line}\nAnswer: {answer_text.strip()}"
        else:
            return f"{q_line}\nAnswer:"

    # ------------------------------------------------------------------
    # Few-shot blocks
    # ------------------------------------------------------------------

    def _mcf_fewshot_block(self) -> str:
        shots = self.fewshot_examples[: self.num_shots]
        if not shots:
            return ""
        blocks = [
            self._build_mcf_body(
                ex["prompt"],
                ex["choices"],
                answer_label=LABELS[ex["correct"]],
            )
            for ex in shots
        ]
        return "\n\n".join(blocks)

    def _cf_fewshot_block(self) -> str:
        shots = self.fewshot_examples[: self.num_shots]
        if not shots:
            return ""
        blocks = [
            self._build_cf_body(
                ex["prompt"],
                answer_text=ex["choices"][ex["correct"]],
            )
            for ex in shots
        ]
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_mcf(self, row: Dict, question_idx: int = 0) -> MCFFormattedRow:
        """
        Format a dataset row for MCF evaluation.

        Choices are shuffled deterministically per question (seed + question_idx)
        so the correct answer is not always at position A.
        """
        choices = [str(row["target"]), str(row["false1"]), str(row["false2"]), str(row["false3"])]

        if self.shuffle_choices:
            rng = random.Random(self.seed + question_idx)
            indices = list(range(4))
            rng.shuffle(indices)
            shuffled = [choices[i] for i in indices]
            # target was at index 0; find where it ended up after shuffle
            correct_pos = indices.index(0)
            correct_label = LABELS[correct_pos]
        else:
            shuffled = choices
            correct_label = "A"  # target always at A when no shuffle

        fewshot = self._mcf_fewshot_block()
        test_body = self._build_mcf_body(row["prompt"], shuffled)
        full_prompt = f"{fewshot}\n\n{test_body}" if fewshot else test_body

        return MCFFormattedRow(
            prompt=full_prompt,
            correct_label=correct_label,
            label_to_choice={l: c for l, c in zip(LABELS, shuffled)},
        )

    def format_cf(self, row: Dict) -> CFFormattedRow:
        """Format a dataset row for CF evaluation."""
        fewshot = self._cf_fewshot_block()
        test_body = self._build_cf_body(row["prompt"])
        full_prompt = f"{fewshot}\n\n{test_body}" if fewshot else test_body

        # Trailing space so first continuation token is unambiguous
        if not full_prompt.endswith(" "):
            full_prompt = full_prompt + " "

        def _lead(s: str) -> str:
            s = str(s).strip()
            return (" " + s) if s else ""

        return CFFormattedRow(
            prompt=full_prompt,
            target=_lead(row["target"]),
            false1=_lead(row["false1"]),
            false2=_lead(row["false2"]),
            false3=_lead(row["false3"]),
        )


# =============================================================================
# Loader helpers
# =============================================================================

def load_fewshot_examples(path: str) -> List[Dict]:
    """
    Load few-shot examples from a YAML file.

    Expected format:
        - prompt: "Which bone is the longest in the human body?"
          choices:
            - "Femur"      # A
            - "Tibia"      # B
            - "Humerus"    # C
            - "Fibula"     # D
          correct: 0       # 0-based index of the correct choice
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Few-shot file not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {p}, got {type(data).__name__}")
    for i, ex in enumerate(data):
        for key in ("prompt", "choices", "correct"):
            if key not in ex:
                raise ValueError(f"Few-shot example {i} missing key '{key}' in {p}")
        if len(ex["choices"]) != 4:
            raise ValueError(
                f"Few-shot example {i} must have exactly 4 choices, "
                f"got {len(ex['choices'])} in {p}"
            )
        if not (0 <= ex["correct"] <= 3):
            raise ValueError(
                f"Few-shot example {i} 'correct' must be 0–3, "
                f"got {ex['correct']} in {p}"
            )
    return data


def build_formatter(
    task_prefix: str = "question",
    num_shots: int = 5,
    fewshot_source: str = "",
    shuffle_choices: bool = True,
    seed: int = 42,
) -> OLMESFormatter:
    """Construct an OLMESFormatter, loading few-shot examples when provided."""
    examples: List[Dict] = []
    if fewshot_source and num_shots > 0:
        examples = load_fewshot_examples(fewshot_source)
    return OLMESFormatter(
        task_prefix=task_prefix,
        num_shots=num_shots,
        fewshot_examples=examples,
        shuffle_choices=shuffle_choices,
        seed=seed,
    )
