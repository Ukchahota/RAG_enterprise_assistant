"""
S3 â€” Modular RAG + Corrective hallucination detection.

Pipeline:
    BM25 top-50 -> CrossEncoder rerank -> relevance threshold
    -> citation-aware generation
    -> claim-level NLI verification
    -> corrective action (refuse) when the answer is not adequately grounded

Both the original and the corrected answer are logged, so the effect of the
corrective layer can be measured directly (proposal RQ3).
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.bm25_retriever import BM25Retriever
from src.faithfulness import verify_answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "answers_s3_corrective_rag.csv"

SYSTEM = "S3_corrective_rag"
CANDIDATE_K = 50
TOP_K = 5
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CE_THRESHOLD = -1.0

# Corrective triggers
UNSUPPORTED_RATE_TRIGGER = 0.40
CONTRADICTION_TRIGGER = 1          # any contradicted claim triggers correction
MIN_EVIDENCE_TRIGGER = 1           # fewer than this many chunks -> refuse outright

# Stratified evaluation subset: all 20 multi-document questions plus 20 others.
SUBSET_IDS = [
    "V2Q083", "V2Q084", "V2Q085", "V2Q086", "V2Q087", "V2Q088", "V2Q089",
    "V2Q090", "V2Q091", "V2Q092", "V2Q093", "V2Q094", "V2Q095", "V2Q096",
    "V2Q097", "V2Q098", "V2Q099", "V2Q100", "V2Q101", "V2Q102",
    "V2Q001", "V2Q002", "V2Q003", "V2Q004", "V2Q005", "V2Q006", "V2Q007",
    "V2Q008", "V2Q009", "V2Q011", "V2Q012", "V2Q013", "V2Q014", "V2Q015",
    "V2Q017", "V2Q019", "V2Q028", "V2Q035", "V2Q038", "V2Q044",
]

REFUSAL_TEMPLATE = (
    "I cannot answer this question reliably from the available University "
    "documents.\n\n"
    "The retrieved evidence does not adequately support a complete answer: "
    "{reason}\n\n"
    "Please consult the relevant policy document directly, or contact Student "
    "Services for authoritative guidance."
)


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


def decide_correction(verification, n_evidence):
    """Return (should_correct, reason) based on the verification result."""
    if n_evidence < MIN_EVIDENCE_TRIGGER:
        return True, "no retrieved passage passed the relevance threshold."

    if verification["n_contradicted"] >= CONTRADICTION_TRIGGER:
        n = verification["n_contradicted"]
        return True, (
            f"{n} claim(s) in the draft answer were contradicted by the "
            "retrieved evidence."
        )

    rate = verification["unsupported_claim_rate"]
    if rate > UNSUPPORTED_RATE_TRIGGER:
        return True, (
            f"{rate:.0%} of the claims in the draft answer could not be "
            "verified against the retrieved evidence."
        )

    return False, ""


def main() -> None:
    questions = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig")
    questions = questions.dropna(subset=["question"])
    # questions = questions[questions["question_id"].isin(SUBSET_IDS)]

    print("Loading BM25 index...")
    retriever = BM25Retriever()

    print(f"Loading CrossEncoder: {CE_MODEL}")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CE_MODEL)

    client, model = get_client()

    print(f"\nSystem:    {SYSTEM}")
    print(f"Model:     {model}")
    print(f"Questions: {len(questions)} (stratified subset)")
    print(f"Triggers:  contradiction >= {CONTRADICTION_TRIGGER}, "
          f"unsupported rate > {UNSUPPORTED_RATE_TRIGGER}\n")

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
        below_threshold = evidence.empty

        if evidence.empty:
            evidence = reranked.head(1)

        draft, error = call_model(
            client, model, build_prompt(row["question"], evidence)
        )

        evidence_list = [
            {"chunk_id": str(r["chunk_id"]), "chunk_text": str(r["chunk_text"])}
            for _, r in evidence.iterrows()
        ]
        verification = verify_answer(draft, evidence_list)

        n_eff = 0 if below_threshold else len(evidence)
        should_correct, reason = decide_correction(verification, n_eff)

        final = REFUSAL_TEMPLATE.format(reason=reason) if should_correct else draft
        elapsed = time.perf_counter() - start

        gold = {
            x.strip() for x in str(row.get("gold_chunk_ids", "")).split("|") if x.strip()
        }
        supplied = {e["chunk_id"] for e in evidence_list}

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
            "answer": final,
            "draft_answer": draft,
            "corrective_action": "refused" if should_correct else "accepted",
            "correction_reason": reason,
            "evidence_chunk_ids": " | ".join(e["chunk_id"] for e in evidence_list),
            "evidence_texts": json.dumps(
                [e["chunk_text"] for e in evidence_list], ensure_ascii=False
            ),
            "evidence_scores": " | ".join(f"{s:.3f}" for s in evidence["ce_score"]),
            "n_evidence": len(evidence_list),
            "n_withheld": withheld,
            "gold_in_evidence": len(gold & supplied) / len(gold) if gold else 0.0,
            "draft_faithfulness": verification["faithfulness"],
            "draft_unsupported_rate": verification["unsupported_claim_rate"],
            "draft_citation_accuracy": verification["citation_accuracy"],
            "draft_n_supported": verification["n_supported"],
            "draft_n_unsupported": verification["n_unsupported"],
            "draft_n_contradicted": verification["n_contradicted"],
            "draft_risk": verification["hallucination_risk"],
            "answer_words": len(str(final).split()),
            "has_citations": "[" in str(final) and "]" in str(final),
            "latency_seconds": elapsed,
            "error": error or "",
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    refused = df[df["corrective_action"] == "refused"]
    accepted = df[df["corrective_action"] == "accepted"]

    print("\n" + "=" * 66)
    print(f"S3 â€” CORRECTIVE RAG ({model})")
    print("=" * 66)
    print(f"Questions:              {len(df)}")
    print(f"Answers accepted:       {len(accepted)}")
    print(f"Answers refused:        {len(refused)}")
    print(f"Mean latency:           {df['latency_seconds'].mean():.2f}s")

    print("\n--- EFFECT OF THE CORRECTIVE LAYER ---")
    print(f"Draft unsupported rate (all):       "
          f"{df['draft_unsupported_rate'].mean():.3f}")
    if len(accepted):
        print(f"Unsupported rate after correction:  "
              f"{accepted['draft_unsupported_rate'].mean():.3f}")
    prevented = int(refused["draft_n_unsupported"].sum()
                    + refused["draft_n_contradicted"].sum())
    print(f"Unsupported/contradicted claims withheld: {prevented}")

    print("\n--- CORRECTIVE ACTION BY QUESTION TYPE ---")
    print(
        pd.crosstab(df["question_type"], df["corrective_action"]).to_string()
    )

    print("\n--- DRAFT RISK vs ACTION ---")
    print(pd.crosstab(df["draft_risk"], df["corrective_action"]).to_string())

    if len(refused):
        print("\n--- SAMPLE REFUSAL REASONS ---")
        for reason, n in refused["correction_reason"].value_counts().head(4).items():
            print(f"  [{n}x] {reason}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
