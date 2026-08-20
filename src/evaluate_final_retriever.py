"""
Final frozen retriever evaluation.

Pipeline: BM25 top-50 -> CrossEncoder rerank -> top-k
Evaluated over all answerable V2 questions.
"""

from pathlib import Path
import time

import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.bm25_retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "results" / "retrieval_evaluation_final_reranked.csv"

CANDIDATE_K = 50
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REPORT_KS = [1, 3, 5, 10]


def parse_gold(value: str) -> set[str]:
    return {x.strip() for x in str(value).split("|") if x.strip()}


def main() -> None:
    questions = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig")
    questions = questions.dropna(subset=["question", "gold_chunk_ids"])

    print(f"Loaded {len(questions)} answerable questions.")
    print("Loading BM25 retriever...")
    retriever = BM25Retriever()

    print(f"Loading CrossEncoder: {CE_MODEL}")
    reranker = CrossEncoder(CE_MODEL)

    rows = []

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        gold = parse_gold(row["gold_chunk_ids"])

        start = time.perf_counter()

        pool = retriever.retrieve(row["question"], top_k=CANDIDATE_K).copy()

        pairs = [[row["question"], str(t)] for t in pool["chunk_text"]]
        pool["ce_score"] = reranker.predict(pairs)

        reranked = pool.sort_values("ce_score", ascending=False).reset_index(drop=True)
        ranked_ids = reranked["chunk_id"].astype(str).tolist()

        elapsed = time.perf_counter() - start

        record = {
            "question_id": row["question_id"],
            "question": row["question"],
            "question_type": row.get("question_type", ""),
            "difficulty": row.get("difficulty", ""),
            "gold_documents": row.get("gold_documents", ""),
            "gold_chunk_ids": row["gold_chunk_ids"],
            "gold_chunk_count": len(gold),
            "latency_seconds": elapsed,
            "bm25_hit_at_5": int(
                bool(gold & set(pool["chunk_id"].astype(str).head(5)))
            ),
            "bm25_recall_at_5": len(
                gold & set(pool["chunk_id"].astype(str).head(5))
            ) / len(gold),
            "gold_in_pool": len(gold & set(ranked_ids)) / len(gold),
            "final_chunk_ids": " | ".join(ranked_ids[:10]),
        }

        for k in REPORT_KS:
            found = gold & set(ranked_ids[:k])
            record[f"hit_at_{k}"] = int(bool(found))
            record[f"recall_at_{k}"] = len(found) / len(gold)
            record[f"full_evidence_at_{k}"] = int(len(found) == len(gold))
            record[f"precision_at_{k}"] = len(found) / k

        # Reciprocal rank over the reranked list.
        rr = 0.0
        for position, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in gold:
                rr = 1.0 / position
                break
        record["mrr"] = rr

        rows.append(record)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("FINAL RETRIEVER — BM25 top-50 + CrossEncoder rerank")
    print("=" * 72)
    print(f"Questions: {len(df)}")
    print(f"Mean latency: {df['latency_seconds'].mean():.3f}s")
    print(f"Gold chunks present in top-{CANDIDATE_K} pool: {df['gold_in_pool'].mean():.3f}")

    print("\n--- OVERALL ---")
    print(f"{'k':>3} | {'Hit':>6} | {'Recall':>6} | {'FullEv':>6} | {'Prec':>6}")
    for k in REPORT_KS:
        print(
            f"{k:>3} | {df[f'hit_at_{k}'].mean():>6.3f} | "
            f"{df[f'recall_at_{k}'].mean():>6.3f} | "
            f"{df[f'full_evidence_at_{k}'].mean():>6.3f} | "
            f"{df[f'precision_at_{k}'].mean():>6.3f}"
        )
    print(f"MRR: {df['mrr'].mean():.3f}")

    print("\n--- BY QUESTION TYPE (k=5) ---")
    by_type = df.groupby("question_type").agg(
        n=("question_id", "count"),
        bm25_hit5=("bm25_hit_at_5", "mean"),
        ce_hit5=("hit_at_5", "mean"),
        ce_hit10=("hit_at_10", "mean"),
        ce_recall5=("recall_at_5", "mean"),
        ce_full5=("full_evidence_at_5", "mean"),
    ).round(3)
    by_type["delta_hit5"] = (by_type["ce_hit5"] - by_type["bm25_hit5"]).round(3)
    print(by_type.to_string())

    print("\n--- BY DIFFICULTY (k=5) ---")
    print(
        df.groupby("difficulty").agg(
            n=("question_id", "count"),
            bm25_hit5=("bm25_hit_at_5", "mean"),
            ce_hit5=("hit_at_5", "mean"),
            ce_recall5=("recall_at_5", "mean"),
        ).round(3).to_string()
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()