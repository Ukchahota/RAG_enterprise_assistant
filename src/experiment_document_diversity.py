import pandas as pd
from src.bm25_retriever import BM25Retriever

q = pd.read_csv("data/evaluation/test_questions_v2_answerable.csv", encoding="utf-8-sig")
q = q[q["question_type"].isin(["multi_document_comparison", "multi_document_synthesis"])]

r = BM25Retriever()

def capped(df, cap, k=5):
    kept, counts = [], {}
    for _, row in df.iterrows():
        d = str(row["document_name"])
        if counts.get(d, 0) < cap:
            kept.append(str(row["chunk_id"]))
            counts[d] = counts.get(d, 0) + 1
        if len(kept) == k:
            break
    return set(kept)

rows = []
for _, row in q.iterrows():
    gold = {x.strip() for x in str(row["gold_chunk_ids"]).split("|") if x.strip()}
    deep = r.retrieve(row["question"], top_k=50)
    rec = {"question_id": row["question_id"], "gold_n": len(gold)}
    for cap in [5, 3, 2, 1]:
        sel = capped(deep, cap)
        found = len(gold & sel)
        rec[f"cap{cap}_found"] = found
        rec[f"cap{cap}_recall"] = found / len(gold)
        rec[f"cap{cap}_full"] = int(found == len(gold))
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("results/multidoc_diversity_caps.csv", index=False, encoding="utf-8-sig")

print(df.to_string(index=False))
print("\ncap | any-hit | mean recall | full evidence")
for cap in [5, 3, 2, 1]:
    print(f" {cap}  |  {int((df[f'cap{cap}_found']>0).sum()):2d}/20  |    {df[f'cap{cap}_recall'].mean():.3f}    |    {int(df[f'cap{cap}_full'].sum()):2d}/20")