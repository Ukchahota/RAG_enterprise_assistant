# analyse_correctness.py
import pandas as pd

samp = pd.read_csv(r"results\correctness_sample.csv")
sc   = pd.read_csv(r"results\correctness_scores.csv")
s3   = pd.read_csv(r"results\answers_s3_corrective_rag.csv")

d = samp.merge(sc, on="question_id").merge(
    s3[["question_id", "draft_n_unsupported", "draft_n_contradicted", "gold_in_evidence"]],
    on="question_id")

print(f"n = {len(d)}\n")
print("Correctness:")
print(d["correctness"].value_counts().to_string(), "\n")

answered = d[d["correctness"] != "refused"]
if len(answered):
    acc = (answered["correctness"] == "correct").mean()
    print(f"Correct among answered: {acc:.3f}  (n={len(answered)})\n")

# the key cross-tab: grounded vs correct
ung = d["draft_n_unsupported"].fillna(0) + d["draft_n_contradicted"].fillna(0)
d = d.assign(grounded=["grounded" if x == 0 else "ungrounded" for x in ung])
print("Faithfulness x correctness:")
print(pd.crosstab(d["grounded"], d["correctness"]), "\n")

print("By question type:")
print(pd.crosstab(d["question_type"], d["correctness"]), "\n")

print("By corrective action:")
print(pd.crosstab(d["corrective_action"], d["correctness"]))