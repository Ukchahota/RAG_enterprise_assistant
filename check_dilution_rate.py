import pandas as pd, json
from src.faithfulness import _nli_scores, strip_citations

c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
band = c[(c["label"] == "unsupported")
         & (c["best_entailment"] >= 0.2) & (c["best_entailment"] < 0.5)]
samp = band.sample(min(20, len(band)), random_state=42)

ans = {
    "S1_basic_rag": pd.read_csv(r"results\answers_s1_basic_rag.csv"),
    "S3_corrective_rag": pd.read_csv(r"results\answers_s3_corrective_rag.csv"),
}

rows = []
for _, r in samp.iterrows():
    a = ans[r["system"]]
    m = a[a["question_id"] == r["question_id"]]
    if m.empty:
        continue
    try:
        texts = json.loads(m.iloc[0]["evidence_texts"])
    except Exception:
        continue
    claim = strip_citations(str(r["sentence"]))
    best_full = best_trim = 0.0
    for t in texts:
        t = str(t)
        best_full = max(best_full, _nli_scores(t, claim)[0])
        for n in (300, 400, 500, 600):
            best_trim = max(best_trim, _nli_scores(t[:n], claim)[0])
    rows.append({"qid": r["question_id"], "system": r["system"],
                 "orig": round(r["best_entailment"], 3),
                 "full": round(best_full, 3), "trimmed": round(best_trim, 3),
                 "recovered": best_trim >= 0.5})

d = pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print(f"recovered above 0.5 when premise trimmed: {d['recovered'].sum()}/{len(d)}")
