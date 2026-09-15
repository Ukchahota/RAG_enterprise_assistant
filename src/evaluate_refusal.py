"""
Refusal behaviour on unanswerable and ambiguous questions (proposal 7.4).

Unanswerable: the corpus contains no answer. Correct behaviour is to refuse.
Ambiguous:    the question needs clarification. Refusing or asking is correct.

Metrics:
    refusal accuracy   - proportion of questions correctly refused
    over-answering rate - proportion answered when the system should not have
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.bm25_retriever import BM25Retriever
from src.retriever import load_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_PATH = PROJECT_ROOT / "results" / "refusal_evaluation.csv"

CANDIDATE_K = 50
TOP_K = 5
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CE_THRESHOLD = -1.0
UNSUPPORTED_TRIGGER = 0.40

REFUSAL_MARKERS = [
    "cannot answer", "can't answer", "unable to answer", "do not have",
    "don't have", "not contain", "does not contain", "no information",
    "not specified", "not stated", "not mentioned", "not provided",
    "insufficient", "unclear", "need more information", "please clarify",
    "consult", "contact student services", "i do not know", "i don't know",
]


def looks_like_refusal(text) -> bool:
    low = str(text).lower()
    return any(m in low for m in REFUSAL_MARKERS)


def get_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not found in .env")
    model = os.getenv("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
    return OpenAI(
        api_key=key, base_url="https://integrate.api.nvidia.com/v1"
    ), model


def call_model(client, model, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=700, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content, None
        except Exception as exc:
            if attempt == retries - 1:
                return "", str(exc)
            time.sleep(2 ** attempt)
    return "", "exhausted retries"


def prompt_no_context(question):
    return (
        "You are a University of Liverpool student support assistant.\n\n"
        "Answer the student's question. If you are not certain, say so "
        "explicitly rather than guessing.\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def prompt_with_context(question, evidence):
    context = "\n\n".join(
        f"[{i}] ({r['document_name']}, page {r['page_number']})\n{r['chunk_text']}"
        for i, (_, r) in enumerate(evidence.iterrows(), start=1)
    )
    return (
        "You are a University of Liverpool student support assistant.\n\n"
        "Answer the question using ONLY the numbered evidence below.\n"
        "Rules:\n"
        "- Cite the evidence number in square brackets after each claim.\n"
        "- If the evidence does not contain the answer, say so explicitly.\n"
        "- Do not add information that is not in the evidence.\n\n"
        f"EVIDENCE:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )


def load_questions():
    frames = []
    for fname, category in [
        ("test_questions_v2_unanswerable.csv", "unanswerable"),
        ("test_questions_v2_ambiguous.csv", "ambiguous"),
    ]:
        p = EVAL_DIR / fname
        if p.exists():
            d = pd.read_csv(p, encoding="utf-8-sig")
            d["category"] = category
            frames.append(d)
        else:
            print(f"  warning: {fname} not found")
    if not frames:
        raise SystemExit("No unanswerable or ambiguous question sets found.")
    return pd.concat(frames, ignore_index=True).dropna(subset=["question"])


def main() -> None:
    questions = load_questions()
    print(f"Questions: {len(questions)}")
    print(questions["category"].value_counts().to_string())

    print("\nLoading retrievers...")
    bm25 = BM25Retriever()
    embed_model, index, metadata = load_retriever()

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(CE_MODEL)

    from src.faithfulness import verify_answer

    client, model = get_client()
    rows = []

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        q = row["question"]
        record = {
            "question_id": row["question_id"],
            "question": q,
            "category": row["category"],
            "question_type": row.get("question_type", ""),
        }

        # ---- S0: no retrieval
        a0, _ = call_model(client, model, prompt_no_context(q))
        record["S0_answer"] = a0
        record["S0_refused"] = looks_like_refusal(a0)

        # ---- S1: dense top-5
        emb = embed_model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        _, idxs = index.search(emb, TOP_K)
        ev1 = metadata.iloc[[i for i in idxs[0] if i >= 0]].copy()
        a1, _ = call_model(client, model, prompt_with_context(q, ev1))
        record["S1_answer"] = a1
        record["S1_refused"] = looks_like_refusal(a1)

        # ---- S2: BM25 + CE + threshold
        pool = bm25.retrieve(q, top_k=CANDIDATE_K).copy()
        pool["ce_score"] = reranker.predict(
            [[q, str(t)] for t in pool["chunk_text"]]
        )
        ranked = pool.sort_values("ce_score", ascending=False).reset_index(drop=True)
        cands = ranked.head(TOP_K)
        ev2 = cands[cands["ce_score"] > CE_THRESHOLD]
        below = ev2.empty
        if below:
            ev2 = ranked.head(1)

        a2, _ = call_model(client, model, prompt_with_context(q, ev2))
        record["S2_answer"] = a2
        record["S2_refused"] = looks_like_refusal(a2)
        record["best_ce_score"] = float(ranked.iloc[0]["ce_score"])
        record["below_threshold"] = below

        # ---- S3: S2 + verification + corrective refusal
        ev_list = [
            {"chunk_id": str(r["chunk_id"]), "chunk_text": str(r["chunk_text"])}
            for _, r in ev2.iterrows()
        ]
        v = verify_answer(a2, ev_list)
        corrective_refuse = (
            v["n_contradicted"] > 0
            or v["unsupported_claim_rate"] > UNSUPPORTED_TRIGGER
            or below
        )
        record["S3_refused"] = bool(corrective_refuse or looks_like_refusal(a2))
        record["S3_unsupported_rate"] = v["unsupported_claim_rate"]
        record["S3_risk"] = v["hallucination_risk"]

        rows.append(record)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("REFUSAL BEHAVIOUR")
    print("=" * 70)
    print(f"{'system':<8} {'refusal accuracy':>18} {'over-answering':>16}")
    for s in ["S0", "S1", "S2", "S3"]:
        acc = df[f"{s}_refused"].mean()
        print(f"{s:<8} {acc:>18.3f} {1 - acc:>16.3f}")

    print("\n--- BY CATEGORY ---")
    print(
        df.groupby("category")[
            [f"{s}_refused" for s in ["S0", "S1", "S2", "S3"]]
        ].mean().round(3).to_string()
    )

    print("\n--- QUESTIONS ANSWERED BY S3 (over-answering cases) ---")
    over = df[~df["S3_refused"]]
    if over.empty:
        print("  none - S3 refused every unanswerable/ambiguous question")
    else:
        for _, r in over.iterrows():
            print(f"  {r['question_id']} ({r['category']}) "
                  f"CE {r['best_ce_score']:.2f} - {r['question'][:70]}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()