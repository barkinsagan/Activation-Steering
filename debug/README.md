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
