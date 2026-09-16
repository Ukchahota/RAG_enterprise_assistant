import glob
import pandas as pd

sample_ids = set(pd.read_csv(r"results\correctness_scores.csv")["question_id"])
print(f"Sample questions: {len(sample_ids)}")

files = glob.glob(r"results\**\*s0*.csv", recursive=True)
print("S0 files found:", files or "none")

for f in files:
    df = pd.read_csv(f)
    print(f"\n{f}  shape={df.shape}")
    print("  columns:", df.columns.tolist())
    if "question_id" not in df.columns:
        continue
    sub = df[df["question_id"].isin(sample_ids)]
    print(f"  covers {sub['question_id'].nunique()} of {len(sample_ids)} sample questions")
    if "answer" in df.columns:
        empty = sub["answer"].fillna("").astype(str).str.strip().eq("").sum()
        print(f"  empty answers in sample: {empty}")
    if "error" in df.columns:
        errs = sub["error"].fillna("").astype(str).str.strip().ne("").sum()
        print(f"  errors in sample: {errs}")
