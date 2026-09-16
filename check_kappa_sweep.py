import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score

d = pd.read_csv(r"results\answers_s3_corrective_rag.csv")
best = d["evidence_scores"].astype(str).str.split("|").str[0].astype(float)
gold = d["gold_in_evidence"].fillna(0) >= 1.0
ung  = d["draft_n_unsupported"].fillna(0)          # unsupported only, precision-1.000 signal

gold_attr = np.where(ung == 0, "None", np.where(gold, "Generation", "Retrieval"))
mask = gold_attr != "None"
print("failures:", mask.sum())
maj = pd.Series(gold_attr[mask]).value_counts(normalize=True).max()
print(f"majority baseline: {maj:.3f}\n")

for t in np.arange(1.0, 8.1, 0.5):
    runtime = np.where(best < t, "Retrieval", "Generation")
    k = cohen_kappa_score(gold_attr[mask], runtime[mask])
    agree = (gold_attr[mask] == runtime[mask]).mean()
    print(f"threshold {t:4.1f}  kappa={k:+.3f}  agreement={agree:.3f}")
