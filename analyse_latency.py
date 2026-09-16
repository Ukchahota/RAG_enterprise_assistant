# analyse_latency.py
import pandas as pd

common = pd.read_csv(r"results\common_question_set.csv")["question_id"]

frames = {}
for name, path in [("S0", r"results\answers_s0_llm_only.csv"),
                   ("S1", r"results\answers_s1_basic_rag.csv"),
                   ("S2", r"results\answers_s2_modular_rag.csv"),
                   ("S3", r"results\answers_s3_corrective_rag.csv")]:
    d = pd.read_csv(path)
    d = d[d["question_id"].isin(common)]
    frames[name] = d["latency_seconds"].dropna()

summary = pd.DataFrame({
    "n":      {k: len(v) for k, v in frames.items()},
    "mean":   {k: round(v.mean(), 2) for k, v in frames.items()},
    "median": {k: round(v.median(), 2) for k, v in frames.items()},
    "p95":    {k: round(v.quantile(0.95), 2) for k, v in frames.items()},
    "max":    {k: round(v.max(), 2) for k, v in frames.items()},
})
print(summary.to_string(), "\n")

if "S2" in frames and "S3" in frames:
    print(f"S3 - S2 mean overhead: {frames['S3'].mean() - frames['S2'].mean():.2f}s")
    print(f"S3 / S2 ratio:         {frames['S3'].mean() / frames['S2'].mean():.2f}x")