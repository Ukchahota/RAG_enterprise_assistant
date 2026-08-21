"""
S2 — Modular RAG: BM25 top-50 -> CrossEncoder rerank -> relevance threshold
-> citation-aware generation.

Saves answers with their supplied evidence, in the same schema as S0, so the
faithfulness verifier can consume both.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.bm25_retriever import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "answers_s2_modular_rag.csv"

SYSTEM = "S2_modular_rag"
CANDIDATE_K = 50
TOP_K = 5
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CE_THRESHOLD = -1.0


def get_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not found in .env")

    model = os.getenv("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1")
    return client, model


def build_prompt(question, evidence):
    context = "\n\n".join(
        f"[{i}] ({row['document_name']}, page {row['page_number']})\n{row['chunk_text']}"
        for i, (_, row) in enumerate(evidence.iterrows(), start=1)
    )
    return (
        "You are a University of Liverpool student support assistant.\n\n"
        "Answer the question using ONLY the numbered evidence below.\n"
        "Rules:\n"
        "- Cite the evidence number in square brackets after each claim, e.g. [2].\n"
        "- Cite only the specific evidence that supports that claim. Do not cite "
        "every item after every sentence.\n"
        "- If the evidence does not contain the answer, say so explicitly.\n"
        "- Do not add information that is not in the evidence.\n"
        "- If the question contains a false assumption, correct it.\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def call_model(client, model, prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=900,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content, None
        except Exception as exc:
            if attempt == retries - 1:
                return "", str(exc)
            time.sleep(2 ** attempt)
    return "", "exhausted retries"


def main() -> None:
    questions = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig")
    questions = questions.dropna(subset=["question"])

    print("Loading BM25 index...")
    retriever = BM25Retriever()

    print(f"Loading CrossEncoder: {CE_MODEL}")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CE_MODEL)

    client, model = get_client()

    print(f"\nSystem:    {SYSTEM}")
    print(f"Model:     {model}")
    print(f"Questions: {len(questions)}")
    print(f"Pipeline:  BM25 top-{CANDIDATE_K} -> CE rerank -> "
          f"threshold {CE_THRESHOLD} -> top-{TOP_K}\n")

    rows = []

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        start = time.perf_counter()

        pool = retriever.retrieve(row["question"], top_k=CANDIDATE_K).copy()
        pool["ce_score"] = reranker.predict(
            [[row["question"], str(t)] for t in pool["chunk_text"]]
        )
        reranked = pool.sort_values("ce_score", ascending=False).reset_index(drop=True)

        candidates = reranked.head(TOP_K)
        evidence = candidates[candidates["ce_score"] > CE_THRESHOLD]
        withheld = len(candidates) - len(evidence)

        if evidence.empty:
            evidence = reranked.head(1)

        answer, error = call_model(
            client, model, build_prompt(row["question"], evidence)
        )
        elapsed = time.perf_counter() - start

        gold = {
            x.strip() for x in str(row.get("gold_chunk_ids", "")).split("|") if x.strip()
        }
        supplied = set(evidence["chunk_id"].astype(str))

        rows.append({
            "system": SYSTEM,
            "model": model,
            "question_id": row["question_id"],
            "question": row["question"],
            "question_type": row.get("question_type", ""),
            "difficulty": row.get("difficulty", ""),
            "expected_answer": row.get("expected_answer", ""),
            "gold_documents": row.get("gold_documents", ""),
            "gold_chunk_ids": row.get("gold_chunk_ids", ""),
            "answer": answer,
            "evidence_chunk_ids": " | ".join(evidence["chunk_id"].astype(str)),
            "evidence_texts": json.dumps(
                evidence["chunk_text"].astype(str).tolist(), ensure_ascii=False
            ),
            "evidence_scores": " | ".join(
                f"{s:.3f}" for s in evidence["ce_score"]
            ),
            "n_evidence": len(evidence),
            "n_withheld": withheld,
            "gold_in_evidence": len(gold & supplied) / len(gold) if gold else 0.0,
            "answer_words": len(str(answer).split()),
            "has_citations": "[" in str(answer) and "]" in str(answer),
            "latency_seconds": elapsed,
            "error": error or "",
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    failed = int((df["error"] != "").sum())

    print("\n" + "=" * 60)
    print(f"S2 — MODULAR RAG ({model})")
    print("=" * 60)
    print(f"Answers generated:    {len(df) - failed}/{len(df)}")
    if failed:
        print(f"API failures:         {failed}")
    print(f"Mean evidence chunks: {df['n_evidence'].mean():.2f}")
    print(f"Mean withheld:        {df['n_withheld'].mean():.2f}")
    print(f"Gold in evidence:     {df['gold_in_evidence'].mean():.3f}")
    print(f"Answers with cites:   {int(df['has_citations'].sum())}/{len(df)}")
    print(f"Mean answer length:   {df['answer_words'].mean():.0f} words")
    print(f"Mean latency:         {df['latency_seconds'].mean():.2f}s")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()