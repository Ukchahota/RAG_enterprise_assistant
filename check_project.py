import pandas as pd, faiss
from pathlib import Path

print("=" * 70); print("CORPUS"); print("=" * 70)
m = pd.read_csv("vector_store/chunk_metadata.csv", encoding="utf-8-sig")
idx = faiss.read_index("vector_store/faiss_index.index")
print("Chunks:", len(m), "| Documents:", m["document_name"].nunique())
print("FAISS vectors:", idx.ntotal, "| Aligned:", idx.ntotal == len(m))

print(); print("=" * 70); print("EVALUATION SET"); print("=" * 70)
q = pd.read_csv("data/evaluation/test_questions_v2_master.csv", encoding="utf-8-sig")
print("Total questions:", len(q))
print(q["question_type"].value_counts().to_string())

print(); print("=" * 70); print("RETRIEVAL (BM25 top-50 + CE rerank)"); print("=" * 70)
r = pd.read_csv("results/retrieval_evaluation_final_reranked.csv", encoding="utf-8-sig")
for k in [1, 3, 5, 10]:
    print("  k={:<3} Hit {:.3f} | Recall {:.3f} | FullEv {:.3f}".format(
        k, r["hit_at_" + str(k)].mean(), r["recall_at_" + str(k)].mean(),
        r["full_evidence_at_" + str(k)].mean()))
print("  MRR {:.3f} | Gold in pool {:.3f}".format(
    r["mrr"].mean(), r["gold_in_pool"].mean()))

print(); print("=" * 70); print("SYSTEMS BUILT"); print("=" * 70)
systems = [("S0 LLM-only", "answers_s0_llm_only.csv"),
           ("S1 Basic RAG", "answers_s1_basic_rag.csv"),
           ("S2 Modular RAG", "answers_s2_modular_rag.csv"),
           ("S3 Corrective", "answers_s3_corrective_rag.csv")]
for name, fname in systems:
    p = Path("results") / fname
    if p.exists():
        d = pd.read_csv(p, encoding="utf-8-sig")
        extra = ""
        if "gold_in_evidence" in d.columns:
            extra = " | gold in evidence {:.3f}".format(d["gold_in_evidence"].mean())
        print("  [OK]      {:<16} {} answers{}".format(name, len(d), extra))
    else:
        print("  [MISSING] " + name)

print(); print("=" * 70); print("FAITHFULNESS"); print("=" * 70)
p = Path("results/hallucination_evaluation.csv")
if p.exists():
    h = pd.read_csv(p, encoding="utf-8-sig")
    print(h.groupby("system").agg(
        questions=("question_id", "count"),
        faithfulness=("faithfulness", "mean"),
        unsupported=("unsupported_claim_rate", "mean"),
        citation_acc=("citation_accuracy", "mean")).round(3).to_string())
else:
    print("  not run yet")

print(); print("=" * 70); print("S3 CORRECTIVE LAYER"); print("=" * 70)
p = Path("results/answers_s3_corrective_rag.csv")
if p.exists():
    s3 = pd.read_csv(p, encoding="utf-8-sig")
    ref = s3[s3["corrective_action"] == "refused"]
    acc = s3[s3["corrective_action"] == "accepted"]
    print("  Questions:", len(s3), "| Accepted:", len(acc), "| Refused:", len(ref))
    print("  Unsupported rate: {:.3f} -> {:.3f}".format(
        s3["draft_unsupported_rate"].mean(), acc["draft_unsupported_rate"].mean()))
    print("  Claims withheld:", int(
        ref["draft_n_unsupported"].sum() + ref["draft_n_contradicted"].sum()))
else:
    print("  not run yet")
