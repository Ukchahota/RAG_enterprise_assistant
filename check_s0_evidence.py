import pandas as pd

m = pd.read_csv(r"results\hallucination_evaluation_all4.csv")
print(m.groupby("system")["n_evidence_verified"].describe().round(2).to_string())
print()
a = pd.read_csv(r"results\answers_s0_llm_only.csv")
print("S0 columns:", [c for c in a.columns if "chunk" in c or "gold" in c])
print("sample gold_chunk_ids:", repr(a["gold_chunk_ids"].iloc[0])[:200] if "gold_chunk_ids" in a.columns else "MISSING")
md = pd.read_csv(r"vector_store\chunk_metadata.csv", encoding="utf-8-sig")
print("sample metadata chunk_id:", repr(md["chunk_id"].iloc[0]))
