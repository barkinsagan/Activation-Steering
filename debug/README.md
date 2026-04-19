# debug/

Standalone diagnostics for the steering pipeline. Run these when experiments give
suspicious results — they're fast, small, and isolate one question each.

## Scripts

### `vector_probe.py`
Vector quality diagnostics + generation eyeball test. For a given set of layers,
captures DIM vectors and reports:

- Cosine similarity between pos-mean and neg-mean (≈1 → DIM vector is noise)
- Pos/neg mean norms (>20% difference → vector encodes magnitude, not direction)
- Between-class distance vs within-class variance ratio (<1 → DIM fails)
- Top-k hidden dimensions contributing to the vector
- Baseline / +coef / −coef generations on a probe prompt (eyeball test)

Typical run (<2 min for 3 layers):
```
python debug/vector_probe.py \
  --pos-file data/prompts/pos.txt \
  --neg-file data/prompts/neg.txt \
  --layers 5 15 20 \
  --coef 3 \
  --gen-prompt "The cranial nerve responsible for vision is the" \
  --out debug/out/vector_quality.csv
```

Pass criteria:
- cos_sim < 0.95
- between/within ratio > 0.5
- generations at +coef should visibly differ from baseline (not gibberish)

Fail → the DIM vector isn't capturing your concept. Either prompts are too
diverse, or the concept isn't linearly encoded at this layer.

### `mcf_exp.py`
Reusable MCF steering experiment harness — swap flags, not cells. Pure
print-only: runs the sweep in memory, prints a per-(layer, coef) summary
and an asymmetry table (Δ(+c), Δ(−c), Δ+ + Δ−), writes nothing to disk.
Intended for quick factorial exploration of prompts × token_position ×
normalize before promoting a variant into a proper logged experiment.

Experiment 1 (current prompts, last, normalize):
```
python debug/mcf_exp.py \
  --exp-id e1_normalize \
  --eval-path data/eval/anatomy_sweep.csv \
  --pos-file data/prompts/pos.txt \
  --neg-file data/prompts/neg.txt \
  --layers 12 18 \
  --coefs -2 -1.5 -1 -0.5 -0.25 0.25 0.5 1 1.5 2 \
  --token-position last --normalize \
  --num-questions 200
```

Variants for later experiments — same command, these flags change:
- Exp 2 (length-matched prompts): `--pos-file ... --neg-file ... --exp-id e2_lenmatch`
- Exp 3 (narrow concept):         `--pos-file ... --neg-file ... --exp-id e3_narrow`
- `--no-normalize` to skip vector normalization
- `--token-position mean` to switch to token-mean activations
