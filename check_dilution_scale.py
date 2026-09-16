import pandas as pd

md = pd.read_csv(r"vector_store\chunk_metadata.csv", encoding="utf-8-sig")
L = md["chunk_text"].astype(str).str.len()
print("chunks:", len(md), "| mean len", round(L.mean(),1),
      "| median", int(L.median()), "| min", L.min(), "| max", L.max())
print()

c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
u = c[c["label"] == "unsupported"]
print("unsupported claims:", len(u))
print("of those, best_entailment in 0.2-0.5 (neutral-collapse band):",
      ((u["best_entailment"] >= 0.2) & (u["best_entailment"] < 0.5)).sum())
print("            below 0.2 (genuine non-entailment):",
      (u["best_entailment"] < 0.2).sum())
print()
print(u["best_entailment"].describe().round(3).to_string())
