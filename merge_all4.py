import pandas as pd

a = pd.read_csv(r"results\hallucination_evaluation_S0_S2_backup.csv")
b = pd.read_csv(r"results\hallucination_evaluation_s1_s3.csv")
c = pd.read_csv(r"results\common_question_set.csv")["question_id"]

m = pd.concat([a, b], ignore_index=True)
m = m[m["question_id"].isin(c)]
m.to_csv(r"results\hallucination_evaluation_all4.csv", index=False, encoding="utf-8-sig")

print("rows per system (common set):")
print(m.groupby("system").size().to_string())
print()
print(m.groupby("system").agg(
    questions=("question_id", "count"),
    claims=("n_checkable", "sum"),
    faithfulness=("faithfulness", "mean"),
    unsupported=("unsupported_claim_rate", "mean"),
    citation_acc=("citation_accuracy", "mean"),
).round(3).to_string())
