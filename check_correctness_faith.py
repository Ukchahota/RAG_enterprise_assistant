import pandas as pd

s = pd.read_csv(r"results\hallucination_evaluation_s1_s3.csv")
s = s[s["system"] == "S3_corrective_rag"]
sc = pd.read_csv(r"results\correctness_scores.csv")

m = sc.merge(s[["question_id", "unsupported_claim_rate", "faithfulness"]],
             on="question_id", how="left")
m["grounded"] = m["unsupported_claim_rate"] <= 0.2

print(pd.crosstab(m["correctness"], m["grounded"],
                  rownames=["correctness"], colnames=["grounded"]).to_string())
print()
print(m.groupby("correctness")[["faithfulness", "unsupported_claim_rate"]].mean().round(3).to_string())
