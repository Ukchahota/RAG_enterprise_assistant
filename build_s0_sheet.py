import pandas as pd
from pathlib import Path

out = Path(r"results\correctness_s0_to_label.csv")
if out.exists():
    raise SystemExit(f"{out} already exists - not overwriting")

s1 = pd.read_csv(r"results\correctness_s1_to_label.csv")
print("S1 label counts (should total 29, with 27 correct):")
print(s1["correctness"].value_counts(dropna=False).to_string())

ids = set(pd.read_csv(r"results\correctness_scores.csv")["question_id"])
s0 = pd.read_csv(r"results\answers_s0_llm_only.csv")

sheet = s0[s0["question_id"].isin(ids)][
    ["question_id", "question_type", "difficulty", "question", "expected_answer", "answer"]
].rename(columns={"answer": "s0_answer"})
assert len(sheet) == 29, f"expected 29 rows, got {len(sheet)} (duplicate question_ids?)"

sheet = sheet.sample(frac=1, random_state=42).reset_index(drop=True)
sheet["correctness"] = ""
sheet["notes"] = ""
sheet.to_csv(out, index=False)
print(f"\nWrote {out} with {len(sheet)} rows")
