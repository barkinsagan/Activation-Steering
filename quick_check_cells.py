import pandas as pd
from pathlib import Path

exp = Path("results/exp_20260626_phys_vs_bio_continuous_full")

CELLS = [
    (11, 0.25),
    (12, 0.25),
    (13, 0.25),
    (14, 0.50),
    (16, 0.25),
    (20, 1.00),
    (25, 3.00),
]

dfs = []
for d in sorted((exp / "cf").glob("layer_*")):
    df = pd.read_csv(d / "detailed_wide.csv")
    df["layer"] = int(d.name.split("_")[1])
    dfs.append(df)
cf = pd.concat(dfs, ignore_index=True)

manifest = pd.read_csv(exp / "split_manifest.csv")
cf = cf.merge(
    manifest[["eval_question_id", "split"]].rename(columns={"eval_question_id": "question_id"}),
    on="question_id",
    how="left",
)

print(f"{'cell':>12}  {'split':>11}  {'n':>3}  {'Dc':>9}  {'SE_Dc':>7}  {'Dw':>9}  {'SE_Dw':>7}  {'Dc-Dw':>9}")
print("-" * 80)

for layer, coef in CELLS:
    cell = cf[(cf.layer == layer) & (abs(cf.coef - coef) < 1e-6)]
    label = f"L{layer:02d} {coef:+.2f}"
    for s in ["validation", "test"]:
        sub = cell[cell.split == s]
        if sub.empty:
            print(f"{label:>12}  {s:>11}  (no data)")
            continue
        dc = sub["delta_target_sum_lp"]
        dw = sub["delta_max_wrong_sum_lp"]
        n = len(sub)
        se_dc = dc.std() / n**0.5 if n else float("nan")
        se_dw = dw.std() / n**0.5 if n else float("nan")
        print(
            f"{label:>12}  {s:>11}  {n:>3}  "
            f"{dc.mean():+.4f}  {se_dc:.4f}  "
            f"{dw.mean():+.4f}  {se_dw:.4f}  "
            f"{(dc - dw).mean():+.4f}"
        )
    print()
