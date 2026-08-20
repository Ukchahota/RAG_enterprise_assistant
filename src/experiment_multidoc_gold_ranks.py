import pandas as pd
from src.bm25_retriever import BM25Retriever

q = pd.read_csv("data/evaluation/test_questions_v2_answerable.csv", encoding="utf-8-sig")
q = q[q["question_type"].isin(["multi_document_comparison", "multi_document_synthesis"])]
meta = pd.read_csv("vector_store/chunk_metadata.csv", encoding="utf-8-sig")
doc_of = dict(zip(meta["chunk_id"].astype(str), meta["document_name"].astype(str)))

r = BM25Retriever()
rows = []

for _, row in q.iterrows():
    gold = [x.strip() for x in str(row["gold_chunk_ids"]).split("|") if x.strip()]
    deep = r.retrieve(row["question"], top_k=1395)
    ranked = deep["chunk_id"].astype(str).tolist()
    docs_top50 = set(deep.head(50)["document_name"].astype(str))
    for g in gold:
        rows.append({
            "question_id": row["question_id"],
            "gold_doc": doc_of.get(g, "?"),
            "gold_rank": ranked.index(g) + 1 if g in ranked else None,
            "doc_in_top50": doc_of.get(g, "?") in docs_top50,
        })

df = pd.DataFrame(rows)
df.to_csv("results/multidoc_gold_ranks.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

print("\nGold chunks:", len(df))
for k in [5, 10, 20, 50, 100, 200]:
    print(f"  within top-{k:<3}: {int((df['gold_rank'] <= k).sum()):2d}/40")
print("Gold document present in top-50:", int(df["doc_in_top50"].sum()), "/40")
print("Median gold rank:", df["gold_rank"].median())