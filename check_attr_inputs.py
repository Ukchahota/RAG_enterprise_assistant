import pandas as pd

d = pd.read_csv(r"results\answers_s3_corrective_rag.csv")
best = d["evidence_scores"].astype(str).str.split("|").str[0].astype(float)
print(best.describe().round(3).to_string())
print()
print("negative scores:", (best < 0).sum(), "of", len(best))
print("gold_in_evidence == 1.0:", (d["gold_in_evidence"] == 1.0).sum())
print("gold_in_evidence in (0,1):", ((d["gold_in_evidence"] > 0) & (d["gold_in_evidence"] < 1)).sum())
print("gold_in_evidence == 0:", (d["gold_in_evidence"] == 0).sum())
