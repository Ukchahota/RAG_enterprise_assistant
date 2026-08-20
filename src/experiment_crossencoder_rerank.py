import pandas as pd
from sentence_transformers import CrossEncoder
from src.bm25_retriever import BM25Retriever

q = pd.read_csv("data/evaluation/test_questions_v2_answerable.csv", encoding="utf-8-sig")
q = q[q["question_type"].isin(["multi_document_comparison", "multi_document_synthesis"])]

r = BM25Retriever()
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

rows = []
for _, row in q.iterrows():
    gold = {x.strip() for x in str(row["gold_chunk_ids"]).split("|") if x.strip()}
    pool = r.retrieve(row["question"], top_k=50).copy()
    pool["ce"] = ce.predict([[row["question"], str(t)] for t in pool["chunk_text"]])
    ranked = pool.sort_values("ce", ascending=False)["chunk_id"].astype(str).tolist()

    rec = {"question_id": row["question_id"], "gold_n": len(gold)}
    for k in [5, 10, 20]:
        found = len(gold & set(ranked[:k]))
        rec[f"ce{k}_found"] = found
        rec[f"ce{k}_recall"] = found / len(gold)
        rec[f"ce{k}_full"] = int(found == len(gold))
    rec["bm25_5_found"] = len(gold & set(pool["chunk_id"].astype(str).head(5)))
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("results/multidoc_crossencoder_rerank.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

print(f"\nBM25 top-5 baseline: any {int((df['bm25_5_found']>0).sum())}/20, recall {(df['bm25_5_found']/df['gold_n']).mean():.3f}")
for k in [5, 10, 20]:
    print(f"CE top-{k:<2}: any {int((df[f'ce{k}_found']>0).sum()):2d}/20, "
          f"recall {df[f'ce{k}_recall'].mean():.3f}, full {int(df[f'ce{k}_full'].sum()):2d}/20")