"""
geometry_sweep.py — Per-layer activation geometry analysis.

Captures activations from up to 11 physics/biology datasets in a single
forward-pass sweep (hooks on all requested layers at once), then for each layer:
  - Saves a 2D PCA scatter plot: activation clouds coloured by dataset,
    6 vector arrows drawn from the PCA origin.
  - Prints a cosine similarity table: 11 datasets × 6 vectors.

Vectors computed:
  V_A  DIM(GPQA phys − bio)
  V_B  DIM(MMLU phys − bio)
  V_C  DIM(text phys − bio)          [raw text, no MCQ format]
  V_D  DIM(Wikipedia phys − bio)
  V_E  DIM(arXiv phys − bioRxiv bio)
  V_F  μ(all physics pooled)          [mean only, no subtraction — reference]

Usage (run each set of layers in a separate HPC cell):
    python analysis/geometry_sweep.py \\
        --config configs/exp_20260626_gpqa_phys_vs_bio_llama8b.yaml \\
        --layers 0 4 8 12 \\
        --save results/geometry_sweep/

    python analysis/geometry_sweep.py ... --layers 13 16 20 24
    python analysis/geometry_sweep.py ... --layers 25 28 31

Dataset paths default to standard locations; override any with named flags.
Missing files are skipped gracefully — analysis runs with available datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from hook import ModelWithHooks


# =============================================================================
# Dataset and vector registries
# =============================================================================

# (key, display_name, is_physics, hex_color, marker)
_DS_SPECS = [
    ("gpqa_phys",  "GPQA physics",         True,  "#E63946", "o"),
    ("gpqa_bio",   "GPQA biology",         False, "#457B9D", "^"),
    ("mmlu_phys",  "MMLU physics",         True,  "#F4A261", "o"),
    ("mmlu_bio",   "MMLU biology",         False, "#2A9D8F", "^"),
    ("text_phys",  "Text physics",         True,  "#C9A800", "o"),
    ("text_bio",   "Text biology",         False, "#264653", "^"),
    ("wiki_phys",  "Wikipedia physics",    True,  "#9B5DE5", "o"),
    ("wiki_bio",   "Wikipedia biology",    False, "#F15BB5", "^"),
    ("arxiv_phys", "arXiv physics",        True,  "#00BBF9", "o"),
    ("arxiv_bio",  "bioRxiv biology",      False, "#00C49A", "^"),
    ("all_phys",   "All physics (pooled)", True,  "#FF4444", "D"),
]
DS_KEYS   = [s[0] for s in _DS_SPECS]
DS_NAME   = {s[0]: s[1] for s in _DS_SPECS}
DS_COLOR  = {s[0]: s[3] for s in _DS_SPECS}
DS_MARKER = {s[0]: s[4] for s in _DS_SPECS}

# (key, display_name, pos_key, neg_key)  neg_key=None → mean only
_VEC_SPECS = [
    ("V_A", "DIM(GPQA phys−bio)",  "gpqa_phys",  "gpqa_bio"),
    ("V_B", "DIM(MMLU phys−bio)",  "mmlu_phys",  "mmlu_bio"),
    ("V_C", "DIM(Text phys−bio)",  "text_phys",  "text_bio"),
    ("V_D", "DIM(Wiki phys−bio)",  "wiki_phys",  "wiki_bio"),
    ("V_E", "DIM(arXiv−bioRxiv)",  "arxiv_phys", "arxiv_bio"),
    ("V_F", "μ(all physics)",      "all_phys",   None),
]
VEC_KEYS  = [v[0] for v in _VEC_SPECS]
VEC_NAME  = {v[0]: v[1] for v in _VEC_SPECS}
VEC_POS   = {v[0]: v[2] for v in _VEC_SPECS}
VEC_NEG   = {v[0]: v[3] for v in _VEC_SPECS}
VEC_COLOR = {
    "V_A": "#E63946",
    "V_B": "#F4A261",
    "V_C": "#C9A800",
    "V_D": "#9B5DE5",
    "V_E": "#00BBF9",
    "V_F": "#888888",
}


# =============================================================================
# Prompt loading
# =============================================================================

def _load_csv(path: str, n: int, seed: int) -> List[str]:
    from olmes.formatter import build_formatter
    df = pd.read_csv(path)
    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    fmt = build_formatter(task_prefix="question", num_shots=0,
                          shuffle_choices=True, seed=seed)
    return [fmt.format_mcf(row, question_idx=i).prompt
            for i, (_, row) in enumerate(df.iterrows())]


def _load_txt(path: str, n: int) -> List[str]:
    lines = Path(path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def load_prompts(path: str, n: int, seed: int) -> Optional[List[str]]:
    """Load up to n prompts from a CSV or TXT file. Returns None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        prompts = _load_csv(path, n, seed) if p.suffix == ".csv" else _load_txt(path, n)
        return prompts or None
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


# =============================================================================
# Multi-layer activation capture
# =============================================================================

def capture_all_layers(
    model_with_hooks: ModelWithHooks,
    tokenizer,
    prompts: List[str],
    layer_names: List[str],
    token_position: str = "last",
    batch_size: int = 16,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """
    Run `prompts` through the model once, collecting last-token activations
    at every layer in `layer_names` in a single sweep.

    Returns {layer_name: [per-prompt [H] tensor on CPU]}.
    """
    mwh = model_with_hooks
    mwh.hook_manager.remove_hooks()
    mwh.register_hooks_on_layers(layer_names)
    mwh.reset_steering()
    mwh.hook_manager.enable()

    result: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    device = next(mwh.model.parameters()).device

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt",
                            max_length=max_length, truncation=True, padding=True)
            ids   = enc["input_ids"].to(device)
            mask  = enc["attention_mask"].to(device)

            mwh.clear_activations()
            mwh(ids, attention_mask=mask)

            for ln in layer_names:
                acts_list = mwh.get_activations(ln)
                if not acts_list:
                    continue
                raw = acts_list[0]  # [B, S, H]

                if token_position == "last":
                    last_idx = (mask.sum(-1) - 1).clamp(0)
                    b_idx    = torch.arange(raw.size(0), device=raw.device)
                    extracted = raw[b_idx, last_idx, :]          # [B, H]
                else:  # mean
                    m = mask.unsqueeze(-1).float()
                    extracted = (raw * m).sum(1) / m.sum(1).clamp(1)

                for vec in extracted.detach().cpu():
                    result[ln].append(vec)

    mwh.hook_manager.remove_hooks()
    mwh.hook_manager.disable()
    return result


# =============================================================================
# Vector computation
# =============================================================================

def compute_vectors(
    layer_acts: Dict[str, Optional[List[torch.Tensor]]],
) -> Dict[str, Optional[torch.Tensor]]:
    """
    For a single layer's activations dict {ds_key: [H-tensors] or None},
    compute all 6 unit-normed steering vectors.
    Returns {vec_key: unit-normed [H] tensor, or None if inputs missing}.
    """
    vectors: Dict[str, Optional[torch.Tensor]] = {}
    for vk in VEC_KEYS:
        pos_acts = layer_acts.get(VEC_POS[vk])
        neg_key  = VEC_NEG[vk]

        if not pos_acts:
            vectors[vk] = None
            continue

        mu_pos = torch.stack(pos_acts).float().mean(0)

        if neg_key is None:
            # V_F: raw mean, no subtraction
            v = F.normalize(mu_pos, dim=0)
        else:
            neg_acts = layer_acts.get(neg_key)
            if not neg_acts:
                vectors[vk] = None
                continue
            mu_neg = torch.stack(neg_acts).float().mean(0)
            v = F.normalize(mu_pos - mu_neg, dim=0)

        vectors[vk] = v

    return vectors


# =============================================================================
# PCA scatter plot
# =============================================================================

def plot_layer(
    layer_idx: int,
    layer_acts: Dict[str, Optional[List[torch.Tensor]]],
    vectors: Dict[str, Optional[torch.Tensor]],
    save_path: Path,
) -> None:
    # ── Collect all activations for joint PCA fit ──
    all_mats, all_keys = [], []
    for key in DS_KEYS:
        acts = layer_acts.get(key)
        if not acts:
            continue
        mat = torch.stack(acts).float().numpy()
        all_mats.append(mat)
        all_keys.extend([key] * len(mat))

    if not all_mats:
        print(f"  [SKIP] Layer {layer_idx}: no activations available")
        return

    X    = np.concatenate(all_mats, axis=0)   # [N_total, H]
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)               # [N_total, 2]

    fig, ax = plt.subplots(figsize=(14, 10))

    # ── Scatter activation clouds ──
    offset = 0
    for key in DS_KEYS:
        acts = layer_acts.get(key)
        if not acts:
            continue
        n  = len(acts)
        xy = X_2d[offset:offset + n]
        offset += n

        ax.scatter(xy[:, 0], xy[:, 1],
                   c=DS_COLOR[key], marker=DS_MARKER[key],
                   alpha=0.45, s=28, edgecolors="none",
                   label=DS_NAME[key])
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        ax.scatter(cx, cy, c=DS_COLOR[key], s=160, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=6)

    # ── Project vectors into PCA space and draw arrows ──
    W      = pca.components_                          # [2, H]
    spread = max(np.ptp(X_2d[:, 0]), np.ptp(X_2d[:, 1]))
    scale  = spread * 0.32

    for vk in VEC_KEYS:
        vec = vectors.get(vk)
        if vec is None:
            continue
        v_2d  = W @ vec.float().numpy()              # [2]
        norm  = np.linalg.norm(v_2d)
        if norm < 1e-8:
            continue
        v_scl = v_2d / norm * scale
        color = VEC_COLOR[vk]

        ax.annotate(
            "", xy=(v_scl[0], v_scl[1]), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5,
                            mutation_scale=20),
            zorder=10,
        )
        ax.text(v_scl[0] * 1.13, v_scl[1] * 1.13,
                f"{vk}\n{VEC_NAME[vk]}",
                color=color, fontsize=7.5, fontweight="bold",
                ha="center", va="center")

    ax.scatter(0, 0, c="black", s=70, marker="+", zorder=11)

    pct1 = pca.explained_variance_ratio_[0]
    pct2 = pca.explained_variance_ratio_[1]
    ax.set_title(
        f"Layer {layer_idx}  —  Activation PCA\n"
        f"PC1 = {pct1:.1%}    PC2 = {pct2:.1%}",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.75)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path.name}")


# =============================================================================
# Cosine similarity table
# =============================================================================

def print_cos_table(
    layer_idx: int,
    layer_acts: Dict[str, Optional[List[torch.Tensor]]],
    vectors: Dict[str, Optional[torch.Tensor]],
) -> None:
    rows = []
    for key in DS_KEYS:
        acts = layer_acts.get(key)
        if not acts:
            continue
        mu   = torch.stack(acts).float().mean(0)
        mu_u = F.normalize(mu, dim=0)
        row  = {"dataset": DS_NAME[key]}
        for vk in VEC_KEYS:
            vec = vectors.get(vk)
            if vec is None:
                row[vk] = float("nan")
            else:
                row[vk] = round(
                    F.cosine_similarity(mu_u.unsqueeze(0), vec.unsqueeze(0)).item(), 4
                )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("dataset")

    header_line = "  " + "   ".join(f"{k}={VEC_NAME[k]}" for k in VEC_KEYS)
    sep = "=" * max(90, len(header_line) + 2)

    print(f"\n{sep}")
    print(f"COSINE SIMILARITY TABLE  —  Layer {layer_idx}")
    print(header_line)
    print(sep)
    with pd.option_context(
        "display.float_format", lambda x: f"{x:+.4f}",
        "display.max_columns", None,
        "display.width", 220,
    ):
        print(df.to_string())
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True,
                    help="Experiment YAML (model settings only)")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="Layer indices to analyse, e.g. --layers 0 4 8 12")
    ap.add_argument("--n",    type=int, default=100,
                    help="Max prompts per dataset (default 100)")
    ap.add_argument("--save", type=Path, default=Path("results/geometry_sweep"),
                    help="Output directory for plots")
    ap.add_argument("--seed", type=int, default=42)

    # Dataset path overrides (all optional — defaults to standard paths)
    ap.add_argument("--gpqa_phys",  default="data/eval/gpqa_main_physics_sweep.csv")
    ap.add_argument("--gpqa_bio",   default="data/eval/gpqa_main_biology_sweep.csv")
    ap.add_argument("--mmlu_phys",  default="data/eval/mmlu_physics_sweep.csv")
    ap.add_argument("--mmlu_bio",   default="data/eval/mmlu_biology_sweep.csv")
    ap.add_argument("--text_phys",  default="data/prompts/physics_pos.txt")
    ap.add_argument("--text_bio",   default="data/prompts/biology_neg.txt")
    ap.add_argument("--wiki_phys",  default="data/eval/wikipedia_physics.txt")
    ap.add_argument("--wiki_bio",   default="data/eval/wikipedia_biology.txt")
    ap.add_argument("--arxiv_phys", default="data/eval/arxiv_physics.txt")
    ap.add_argument("--arxiv_bio",  default="data/eval/biorxiv_biology.txt")
    args = ap.parse_args()

    # ── Read model settings from config YAML (bypass full validation) ──
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    model_raw   = raw["model"]
    sweep_raw   = raw.get("sweep", {})
    layer_pat   = sweep_raw.get("layer_name_pattern", "model.layers.{layer_idx}")
    tok_pos     = sweep_raw.get("token_position", "last")
    cap_bs      = sweep_raw.get("capture_batch_size", 16)

    dtype_map   = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype       = dtype_map[model_raw.get("dtype", "bfloat16")]
    model_name  = model_raw["name"]
    device      = model_raw.get("device", "cuda")

    # ── Load model ──
    print(f"\nLoading model: {model_name}  (dtype={model_raw.get('dtype')}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    mwh = ModelWithHooks(model)

    # ── Build layer names ──
    layer_names = [layer_pat.format(layer_idx=i) for i in args.layers]
    print(f"Layers to analyse: {args.layers}")
    print(f"Layer names: {layer_names}")

    # ── Load prompts for each dataset ──
    ds_path_args = {
        "gpqa_phys":  args.gpqa_phys,
        "gpqa_bio":   args.gpqa_bio,
        "mmlu_phys":  args.mmlu_phys,
        "mmlu_bio":   args.mmlu_bio,
        "text_phys":  args.text_phys,
        "text_bio":   args.text_bio,
        "wiki_phys":  args.wiki_phys,
        "wiki_bio":   args.wiki_bio,
        "arxiv_phys": args.arxiv_phys,
        "arxiv_bio":  args.arxiv_bio,
    }
    print("\nLoading datasets:")
    ds_prompts: Dict[str, Optional[List[str]]] = {}
    for key, path in ds_path_args.items():
        prompts = load_prompts(path, args.n, args.seed)
        ds_prompts[key] = prompts
        status = f"{len(prompts)} prompts" if prompts else "MISSING"
        print(f"  {key:12s}  {status}  ({path})")

    # Build "all physics pooled" from all physics sources
    all_phys_prompts: List[str] = []
    for key in ["gpqa_phys", "mmlu_phys", "text_phys", "wiki_phys", "arxiv_phys"]:
        if ds_prompts.get(key):
            all_phys_prompts.extend(ds_prompts[key])
    ds_prompts["all_phys"] = all_phys_prompts if all_phys_prompts else None
    print(f"  {'all_phys':12s}  {len(all_phys_prompts)} prompts  (pooled)")

    # ── Capture activations for each dataset (single forward-pass sweep per dataset) ──
    # acts_store[ds_key][layer_name] = List[torch.Tensor([H])]
    print("\nCapturing activations (all layers in one sweep per dataset)...")
    acts_store: Dict[str, Optional[Dict[str, List[torch.Tensor]]]] = {}
    for key in DS_KEYS:
        prompts = ds_prompts.get(key)
        if not prompts:
            acts_store[key] = None
            continue
        print(f"  {key} ({len(prompts)} prompts)...")
        acts_store[key] = capture_all_layers(
            mwh, tokenizer, prompts, layer_names,
            token_position=tok_pos, batch_size=cap_bs,
        )

    # ── Per-layer analysis ──
    for layer_idx, layer_name in zip(args.layers, layer_names):
        print(f"\n{'─'*60}")
        print(f"LAYER {layer_idx}  ({layer_name})")

        # Build per-layer activation dict: {ds_key: [H-tensors] or None}
        layer_acts: Dict[str, Optional[List[torch.Tensor]]] = {}
        for key in DS_KEYS:
            if acts_store.get(key) is None:
                layer_acts[key] = None
            else:
                layer_acts[key] = acts_store[key].get(layer_name)

        # Compute 6 vectors
        vectors = compute_vectors(layer_acts)
        present = [vk for vk in VEC_KEYS if vectors.get(vk) is not None]
        missing = [vk for vk in VEC_KEYS if vectors.get(vk) is None]
        print(f"  Vectors computed: {present}")
        if missing:
            print(f"  Vectors skipped (missing data): {missing}")

        # PCA scatter plot
        plot_layer(
            layer_idx, layer_acts, vectors,
            save_path=args.save / f"layer_{layer_idx:02d}.png",
        )

        # Cosine similarity table
        print_cos_table(layer_idx, layer_acts, vectors)

    print("\nDone.")


if __name__ == "__main__":
    main()
