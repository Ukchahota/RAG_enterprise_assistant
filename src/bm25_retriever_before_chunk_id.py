from pathlib import Path
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_chunks" / "document_chunks.csv"


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Converts text to lowercase and keeps only words/numbers.
    """
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def load_chunks() -> pd.DataFrame:
    """
    Load document chunks created by src/chunker.py.
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks_df = pd.read_csv(CHUNKS_PATH, encoding="utf-8-sig")

    chunks_df.columns = (
        chunks_df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required_columns = [
        "chunk_id",
        "document_name",
        "document_type",
        "source",
        "source_url",
        "file_name",
        "page_number",
        "chunk_text",
    ]

    missing_columns = [col for col in required_columns if col not in chunks_df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in document_chunks.csv: {missing_columns}\n"
            f"Found columns: {chunks_df.columns.tolist()}"
        )

    chunks_df = chunks_df.dropna(subset=["chunk_text"]).reset_index(drop=True)

    return chunks_df


class BM25Retriever:
    """
    BM25 keyword retriever over the existing document chunks.
    """

    def __init__(self):
        self.metadata = load_chunks()

        corpus_texts = self.metadata["chunk_text"].astype(str).tolist()
        tokenized_corpus = [tokenize(text) for text in corpus_texts]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.metadata.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        results["rank"] = range(1, len(results) + 1)

        return results[
            [
                "rank",
                "score",
                "document_name",
                "document_type",
                "page_number",
                "chunk_text",
                "source_url",
            ]
        ]


def retrieve_bm25(query: str, top_k: int = 5) -> pd.DataFrame:
    retriever = BM25Retriever()
    return retriever.retrieve(query, top_k=top_k)


def main():
    query = input("Enter your question: ")

    retriever = BM25Retriever()
    results = retriever.retrieve(query, top_k=5)

    print("\nTop BM25 retrieved chunks:\n")

    for _, row in results.iterrows():
        print("=" * 80)
        print(f"Rank: {row['rank']}")
        print(f"Score: {row['score']:.4f}")
        print(f"Document: {row['document_name']}")
        print(f"Type: {row['document_type']}")
        print(f"Page: {row['page_number']}")
        print(f"Source: {row['source_url']}")
        print("\nChunk:")
        print(str(row["chunk_text"])[:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()