import pandas as pd

s = pd.read_csv(r"results\hallucination_evaluation_s1_s3.csv")
a = pd.read_csv(r"results\answers_s3_corrective_rag.csv")[
    ["question_id", "corrective_action", "draft_unsupported_rate"]
]
c = pd.read_csv(r"results\common_question_set.csv")["question_id"]

m = s.merge(a, on="question_id", how="left")
m = m[m["question_id"].isin(c)]
print("rows in common set:", len(m))

print(m.groupby(["system", "corrective_action"])[
    ["unsupported_claim_rate", "faithfulness", "n_checkable"]
].agg(["mean", "count"]).round(3).to_string())

ans = m[(m["system"] == "S3_corrective_rag") & (m["corrective_action"] == "accepted")]
print()
print("S3 accepted only: unsupported", round(ans["unsupported_claim_rate"].mean(), 4),
      "| faithfulness", round(ans["faithfulness"].mean(), 4),
      "| n", len(ans))
print("S3 draft unsupported (all):",
      round(m[m["system"] == "S3_corrective_rag"]["draft_unsupported_rate"].mean(), 4))
