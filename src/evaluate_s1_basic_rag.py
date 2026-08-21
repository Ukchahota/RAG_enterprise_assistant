"""
S1 - Basic RAG: dense FAISS retrieval, top-5, generation.

No hybrid retrieval, no reranking, no relevance thresholding. This is the
naive RAG baseline against which S2 (Modular RAG) is compared for RQ2.

Output schema matches S0/S2/S3 so the faithfulness verifier can consume it.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.retriever import load_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "answers_s1_basic_rag.csv"

SYSTEM = "S1_basic_rag"
TOP_K = 5


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


def dense_retrieve(query, model, index, metadata, top_k=5):
    embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(embedding, top_k)

    valid = [i for i in indices[0] if i >= 0]
    valid_scores = [s for s, i in zip(scores[0], indices[0]) if i >= 0]

    results = metadata.iloc[valid].copy()
    results["score"] = valid_scores
    return results.reset_index(drop=True)


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
        "- Cite only the specific evidence that supports that claim.\n"
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

    print("Loading dense FAISS retriever...")
    embed_model, index, metadata = load_retriever()

    client, model = get_client()

    print(f"\nSystem:    {SYSTEM}")
    print(f"Model:     {model}")
    print(f"Questions: {len(questions)}")
    print(f"Pipeline:  dense FAISS top-{TOP_K} (no rerank, no threshold)\n")

    rows = []

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        start = time.perf_counter()

        evidence = dense_retrieve(
            row["question"], embed_model, index, metadata, TOP_K
        )

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
            "evidence_scores": " | ".join(f"{s:.3f}" for s in evidence["score"]),
            "n_evidence": len(evidence),
            "n_withheld": 0,
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
    print(f"S1 - BASIC RAG ({model})")
    print("=" * 60)
    print(f"Answers generated:   {len(df) - failed}/{len(df)}")
    if failed:
        print(f"API failures:        {failed}")
    print(f"Gold in evidence:    {df['gold_in_evidence'].mean():.3f}")
    print(f"Answers with cites:  {int(df['has_citations'].sum())}/{len(df)}")
    print(f"Mean answer length:  {df['answer_words'].mean():.0f} words")
    print(f"Mean latency:        {df['latency_seconds'].mean():.2f}s")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()