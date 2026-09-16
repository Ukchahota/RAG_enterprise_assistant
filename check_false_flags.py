import pandas as pd

s = pd.read_csv(r"results\hallucination_evaluation_s1_s3.csv")
s = s[s["system"] == "S3_corrective_rag"]
sc = pd.read_csv(r"results\correctness_scores.csv")
a = pd.read_csv(r"results\answers_s3_corrective_rag.csv")

m = sc.merge(s[["question_id", "unsupported_claim_rate", "n_unsupported",
                "n_checkable", "gold_in_evidence"]], on="question_id", how="left")
odd = m[(m["correctness"] == "correct") & (m["unsupported_claim_rate"] > 0.2)]

print(odd.to_string(index=False))
print()
for qid in odd["question_id"]:
    r = a[a["question_id"] == qid].iloc[0]
    print("=" * 70)
    print(qid, "|", r["question_type"], "| gold_in_evidence:", r["gold_in_evidence"])
    print("Q:", str(r["question"])[:200])
    print("-" * 70)
    print("ANSWER:", str(r["answer"])[:700])
