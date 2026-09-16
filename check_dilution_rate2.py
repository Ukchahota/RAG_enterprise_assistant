import pandas as pd, sys
from src.faithfulness import _nli_scores, strip_citations
from src.evaluate_hallucination import load_chunk_texts

ct = load_chunk_texts()
c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
band = c[(c["label"] == "unsupported")
         & (c["best_entailment"] >= 0.2) & (c["best_entailment"] < 0.5)
         & (c["supporting_chunk"].notna())]
samp = band.sample(min(20, len(band)), random_state=42)
print(f"scoring {len(samp)} claims x 2 trims = {len(samp)*2} passes\n", flush=True)

rows = []
for i, (_, r) in enumerate(samp.iterrows(), 1):
    txt = ct.get(str(r["supporting_chunk"]), "")
    if not txt:
        continue
    claim = strip_citations(str(r["sentence"]))
    best = max(_nli_scores(txt[:n], claim)[0] for n in (400, 600))
    rows.append({"qid": r["question_id"], "system": r["system"],
                 "orig": round(r["best_entailment"], 3),
                 "trimmed": round(best, 3), "recovered": best >= 0.5})
    print(f"{i}/{len(samp)} {r['question_id']} {r['best_entailment']:.3f} -> {best:.3f}", flush=True)

d = pd.DataFrame(rows)
print()
print(d.to_string(index=False))
print(f"\nrecovered above 0.5 when trimmed: {d['recovered'].sum()}/{len(d)}")
