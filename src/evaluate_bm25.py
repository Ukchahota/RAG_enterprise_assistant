from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

from src.bm25_retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_questions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = OUTPUT_DIR / "retrieval_evaluation_bm25.csv"

TOP_K = 5


def normalise(text: str) -> str:
    return str(text).strip().lower()


def parse_expected_documents(expected_documents: str) -> list[str]:
    return [normalise(doc) for doc in str(expected_documents).split("||")]


def is_relevant(retrieved_document: str, expected_documents: list[str]) -> bool:
    retrieved_norm = normalise(retrieved_document)

    for expected in expected_documents:
        if expected in retrieved_norm or retrieved_norm in expected:
            return True

    return False


def load_questions() -> pd.DataFrame:
    questions_df = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig")

    questions_df.columns = (
        questions_df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required_columns = [
        "question",
        "expected_documents",
        "question_type",
        "notes",
    ]

    missing_columns = [
        col for col in required_columns if col not in questions_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in test_questions.csv: {missing_columns}\n"
            f"Found columns: {questions_df.columns.tolist()}"
        )

    questions_df = questions_df.dropna(subset=["question", "expected_documents"])

    return questions_df


def reciprocal_rank(results: pd.DataFrame, expected_documents: list[str]) -> float:
    for _, row in results.iterrows():
        if is_relevant(row["document_name"], expected_documents):
            return 1 / row["rank"]

    return 0.0


def precision_at_k(
    results: pd.DataFrame,
    expected_documents: list[str],
    k: int,
) -> float:
    top_results = results.head(k)

    relevant_count = sum(
        is_relevant(row["document_name"], expected_documents)
        for _, row in top_results.iterrows()
    )

    return relevant_count / k


def hit_at_k(
    results: pd.DataFrame,
    expected_documents: list[str],
    k: int,
) -> bool:
    top_results = results.head(k)

    return any(
        is_relevant(row["document_name"], expected_documents)
        for _, row in top_results.iterrows()
    )


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Test questions file not found: {QUESTIONS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions_df = load_questions()

    print("Loading BM25 retriever...")
    retriever = BM25Retriever()

    rows = []

    for _, question_row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        question = question_row["question"]

        expected_documents = parse_expected_documents(
            question_row["expected_documents"]
        )

        start_time = time.time()

        results = retriever.retrieve(
            query=question,
            top_k=TOP_K,
        )

        retrieval_time = time.time() - start_time

        top_1_document = results.iloc[0]["document_name"]
        top_1_score = results.iloc[0]["score"]

        row = {
            "question": question,
            "expected_documents": question_row["expected_documents"],
            "top_1_document": top_1_document,
            "top_1_score": top_1_score,
            "top_1_hit": hit_at_k(results, expected_documents, 1),
            "hit_at_3": hit_at_k(results, expected_documents, 3),
            "hit_at_5": hit_at_k(results, expected_documents, 5),
            "precision_at_5": precision_at_k(results, expected_documents, 5),
            "mrr_at_5": reciprocal_rank(results, expected_documents),
            "retrieval_time_seconds": retrieval_time,
            "retrieved_documents": " | ".join(
                results["document_name"].astype(str).tolist()
            ),
            "retrieved_pages": " | ".join(
                results["page_number"].astype(str).tolist()
            ),
            "question_type": question_row.get("question_type", ""),
            "notes": question_row.get("notes", ""),
        }

        rows.append(row)

    evaluation_df = pd.DataFrame(rows)
    evaluation_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nBM25 Retriever Evaluation")
    print("=" * 50)
    print("Retriever: BM25 keyword search")
    print(f"Total questions: {len(evaluation_df)}")
    print(f"Top-1 Hit Rate: {evaluation_df['top_1_hit'].mean():.2f}")
    print(f"Hit@3: {evaluation_df['hit_at_3'].mean():.2f}")
    print(f"Hit@5: {evaluation_df['hit_at_5'].mean():.2f}")
    print(f"Precision@5: {evaluation_df['precision_at_5'].mean():.2f}")
    print(f"MRR@5: {evaluation_df['mrr_at_5'].mean():.2f}")
    print(
        "Average retrieval time: "
        f"{evaluation_df['retrieval_time_seconds'].mean():.4f} seconds"
    )
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()