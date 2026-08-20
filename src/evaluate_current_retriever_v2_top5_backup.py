from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

from src.retriever import load_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "test_questions_v2_answerable.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "retrieval_evaluation_faiss_v2.csv"
)

TOP_K = 5


def normalise(text: str) -> str:
    return str(text).strip()


def parse_gold_chunk_ids(
    gold_chunk_ids: str,
) -> list[str]:
    """
    Parse gold chunk IDs from the V2 evaluation dataset.

    Multiple gold chunks are separated using |.
    """
    return [
        normalise(chunk_id)
        for chunk_id in str(gold_chunk_ids).split("|")
        if normalise(chunk_id)
    ]


def load_questions() -> pd.DataFrame:
    """
    Load the V2 answerable evaluation questions.
    """
    questions_df = pd.read_csv(
        QUESTIONS_PATH,
        encoding="utf-8-sig",
    )

    questions_df.columns = (
        questions_df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required_columns = [
        "question_id",
        "question",
        "gold_documents",
        "gold_chunk_ids",
        "question_type",
        "difficulty",
        "notes",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in questions_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in V2 question set: "
            f"{missing_columns}\n"
            f"Found columns: "
            f"{questions_df.columns.tolist()}"
        )

    questions_df = questions_df.dropna(
        subset=[
            "question",
            "gold_chunk_ids",
        ]
    )

    questions_df = questions_df[
        questions_df["question"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    return questions_df.reset_index(drop=True)


def retrieve_with_loaded_model(
    query: str,
    model,
    index,
    metadata: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve the top-k chunks using FAISS.
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    valid_indices = [
        index_value
        for index_value in indices[0]
        if index_value >= 0
    ]

    valid_scores = [
        score
        for score, index_value in zip(
            scores[0],
            indices[0],
        )
        if index_value >= 0
    ]

    results = metadata.iloc[
        valid_indices
    ].copy()

    results["score"] = valid_scores

    results["rank"] = range(
        1,
        len(results) + 1,
    )

    required_metadata_columns = [
        "chunk_id",
        "document_name",
        "document_type",
        "page_number",
        "chunk_text",
        "source_url",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in results.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "FAISS metadata is missing columns: "
            f"{missing_metadata_columns}"
        )

    return results[
        [
            "rank",
            "score",
            "chunk_id",
            "document_name",
            "document_type",
            "page_number",
            "chunk_text",
            "source_url",
        ]
    ]


def hit_at_k(
    results: pd.DataFrame,
    gold_chunk_ids: list[str],
    k: int,
) -> bool:
    """
    Return True if at least one gold chunk occurs
    within the top-k retrieved chunks.
    """
    retrieved_ids = set(
        results.head(k)["chunk_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    gold_ids = set(gold_chunk_ids)

    return bool(
        retrieved_ids.intersection(gold_ids)
    )


def reciprocal_rank(
    results: pd.DataFrame,
    gold_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """
    Reciprocal rank of the first exact gold chunk
    found in the top-k results.
    """
    gold_ids = set(gold_chunk_ids)

    for _, row in results.head(k).iterrows():
        retrieved_chunk_id = normalise(
            row["chunk_id"]
        )

        if retrieved_chunk_id in gold_ids:
            return 1.0 / float(
                row["rank"]
            )

    return 0.0


def precision_at_k(
    results: pd.DataFrame,
    gold_chunk_ids: list[str],
    k: int,
) -> float:
    """
    Exact chunk-level precision@k.

    Number of retrieved gold chunks divided by k.
    """
    retrieved_ids = (
        results.head(k)["chunk_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    gold_ids = set(gold_chunk_ids)

    relevant_count = sum(
        chunk_id in gold_ids
        for chunk_id in retrieved_ids
    )

    return relevant_count / k


def recall_at_k(
    results: pd.DataFrame,
    gold_chunk_ids: list[str],
    k: int,
) -> float:
    """
    Fraction of the question's gold chunks retrieved
    within the top-k results.
    """
    gold_ids = set(gold_chunk_ids)

    if not gold_ids:
        return 0.0

    retrieved_ids = set(
        results.head(k)["chunk_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    relevant_retrieved = len(
        retrieved_ids.intersection(
            gold_ids
        )
    )

    return (
        relevant_retrieved
        / len(gold_ids)
    )


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Test questions file not found: "
            f"{QUESTIONS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions_df = load_questions()

    print(
        f"Loaded {len(questions_df)} "
        "answerable V2 questions."
    )

    print("Loading FAISS retriever...")

    model, index, metadata = (
        load_retriever()
    )

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and metadata are not aligned: "
            f"{index.ntotal} vectors vs "
            f"{len(metadata)} metadata rows."
        )

    rows = []

    for _, question_row in tqdm(
        questions_df.iterrows(),
        total=len(questions_df),
    ):
        question_id = (
            question_row["question_id"]
        )

        question = (
            question_row["question"]
        )

        gold_chunk_ids = (
            parse_gold_chunk_ids(
                question_row[
                    "gold_chunk_ids"
                ]
            )
        )

        start_time = time.perf_counter()

        results = (
            retrieve_with_loaded_model(
                query=question,
                model=model,
                index=index,
                metadata=metadata,
                top_k=TOP_K,
            )
        )

        retrieval_time = (
            time.perf_counter()
            - start_time
        )

        top_1_chunk_id = (
            results.iloc[0]["chunk_id"]
            if not results.empty
            else ""
        )

        top_1_document = (
            results.iloc[0][
                "document_name"
            ]
            if not results.empty
            else ""
        )

        top_1_score = (
            results.iloc[0]["score"]
            if not results.empty
            else float("nan")
        )

        row = {
            "question_id":
                question_id,

            "question":
                question,

            "question_type":
                question_row.get(
                    "question_type",
                    "",
                ),

            "difficulty":
                question_row.get(
                    "difficulty",
                    "",
                ),

            "gold_documents":
                question_row[
                    "gold_documents"
                ],

            "gold_chunk_ids":
                question_row[
                    "gold_chunk_ids"
                ],

            "gold_chunk_count":
                len(gold_chunk_ids),

            "top_1_chunk_id":
                top_1_chunk_id,

            "top_1_document":
                top_1_document,

            "top_1_score":
                top_1_score,

            "top_1_hit":
                hit_at_k(
                    results,
                    gold_chunk_ids,
                    1,
                ),

            "hit_at_3":
                hit_at_k(
                    results,
                    gold_chunk_ids,
                    3,
                ),

            "hit_at_5":
                hit_at_k(
                    results,
                    gold_chunk_ids,
                    5,
                ),

            "precision_at_5":
                precision_at_k(
                    results,
                    gold_chunk_ids,
                    5,
                ),

            "recall_at_5":
                recall_at_k(
                    results,
                    gold_chunk_ids,
                    5,
                ),

            "mrr_at_5":
                reciprocal_rank(
                    results,
                    gold_chunk_ids,
                    5,
                ),

            "retrieval_time_seconds":
                retrieval_time,

            "retrieved_chunk_ids":
                " | ".join(
                    results[
                        "chunk_id"
                    ]
                    .astype(str)
                    .tolist()
                ),

            "retrieved_documents":
                " | ".join(
                    results[
                        "document_name"
                    ]
                    .astype(str)
                    .tolist()
                ),

            "retrieved_pages":
                " | ".join(
                    results[
                        "page_number"
                    ]
                    .astype(str)
                    .tolist()
                ),

            "notes":
                question_row.get(
                    "notes",
                    "",
                ),
        }

        rows.append(row)

    evaluation_df = pd.DataFrame(
        rows
    )

    evaluation_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nFAISS Retriever Evaluation — V2"
    )

    print("=" * 60)

    print(
        "Retriever: FAISS dense vector search"
    )

    print(
        "Embedding model: "
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        f"Total questions: "
        f"{len(evaluation_df)}"
    )

    print(
        f"Top-1 Hit Rate: "
        f"{evaluation_df['top_1_hit'].mean():.3f}"
    )

    print(
        f"Hit@3: "
        f"{evaluation_df['hit_at_3'].mean():.3f}"
    )

    print(
        f"Hit@5: "
        f"{evaluation_df['hit_at_5'].mean():.3f}"
    )

    print(
        f"Precision@5: "
        f"{evaluation_df['precision_at_5'].mean():.3f}"
    )

    print(
        f"Recall@5: "
        f"{evaluation_df['recall_at_5'].mean():.3f}"
    )

    print(
        f"MRR@5: "
        f"{evaluation_df['mrr_at_5'].mean():.3f}"
    )

    print(
        "Average retrieval time: "
        f"{evaluation_df['retrieval_time_seconds'].mean():.4f} "
        "seconds"
    )

    print(
        f"\nSaved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
