import pandas as pd

c = pd.read_csv(r"results\hallucination_claims.csv")
print("systems in file:", c["system"].unique().tolist())
print()
print(pd.crosstab(c["system"], c["label"]).to_string())
print()
s0 = c[c["system"] == "S0_llm_only"]
print("S0 entailment distribution:")
print(s0["best_entailment"].describe().round(3).to_string())
print()
print("S0 supporting_chunk non-null:", s0["supporting_chunk"].notna().sum(), "of", len(s0))
