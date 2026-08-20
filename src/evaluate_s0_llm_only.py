"""
S0 baseline — LLM-only, no retrieval.

Answers the evaluation questions from the model's parametric knowledge alone.
This is the control condition for RQ1: does retrieval improve groundedness?
"""

import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "results" / "answers_s0_llm_only.csv"

SYSTEM = "S0_llm_only"

REFUSAL_MARKERS = [
    "i do not have", "i don't have", "i cannot", "i can't",
    "not able to", "no access", "unable to", "i do not know",
    "i don't know", "cannot confirm", "would need to check",
    "recommend checking", "consult the",
]


def get_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not found in .env")

    model = os.getenv("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
    client = OpenAI(
        api_key=key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    return client, model


def build_prompt(question: str) -> str:
    return (
        "You are a University of Liverpool student support assistant.\n\n"
        "Answer the student's question.\n"
        "If you are not certain of the answer, say so explicitly rather "
        "than guessing.\n\n"
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


def looks_like_refusal(answer: str) -> bool:
    low = str(answer).lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def main() -> None:
    questions = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig")
    questions = questions.dropna(subset=["question"])

    client, model = get_client()

    print(f"System:    {SYSTEM}")
    print(f"Model:     {model}")
    print(f"Questions: {len(questions)}")
    print("Retrieval: NONE (parametric knowledge only)\n")

    rows = []

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        start = time.perf_counter()
        answer, error = call_model(client, model, build_prompt(row["question"]))
        elapsed = time.perf_counter() - start

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
            "evidence_chunk_ids": "",
            "evidence_texts": "",
            "n_evidence": 0,
            "answer_words": len(str(answer).split()),
            "has_citations": "[" in str(answer) and "]" in str(answer),
            "looks_like_refusal": looks_like_refusal(answer),
            "latency_seconds": elapsed,
            "error": error or "",
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    failed = int((df["error"] != "").sum())

    print("\n" + "=" * 60)
    print(f"S0 — LLM ONLY ({model})")
    print("=" * 60)
    print(f"Answers generated:   {len(df) - failed}/{len(df)}")
    if failed:
        print(f"API failures:        {failed}")
        print(df[df["error"] != ""]["error"].value_counts().head().to_string())
    print(f"Mean answer length:  {df['answer_words'].mean():.0f} words")
    print(f"Mean latency:        {df['latency_seconds'].mean():.2f}s")
    print(f"Hedged / refused:    {int(df['looks_like_refusal'].sum())}/{len(df)}")

    print("\nBy question type:")
    print(
        df.groupby("question_type").agg(
            n=("question_id", "count"),
            mean_words=("answer_words", "mean"),
            hedged=("looks_like_refusal", "sum"),
        ).round(1).to_string()
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()