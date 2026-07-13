from pathlib import Path
import numpy as np
import pandas as pd

from src.retriever import load_retriever
from src.bm25_retriever import BM25Retriever, tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Dense FAISS retrieval
    2. BM25 keyword retrieval

    Ranking is combined using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        print("Loading dense FAISS retriever...")
        self.dense_model, self.faiss_index, self.dense_metadata = load_retriever()

        print("Loading BM25 retriever...")
        self.bm25_retriever = BM25Retriever()

    def dense_retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        query_embedding = self.dense_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = self.dense_metadata.iloc[indices[0]].copy()
        results["dense_score"] = scores[0]
        results["dense_rank"] = range(1, len(results) + 1)

        return results[
            [
                "chunk_id",
                "document_name",
                "document_type",
                "page_number",
                "chunk_text",
                "source_url",
                "dense_score",
                "dense_rank",
            ]
        ]

    def bm25_retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        query_tokens = tokenize(query)

        scores = self.bm25_retriever.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.bm25_retriever.metadata.iloc[top_indices].copy()
        results["bm25_score"] = scores[top_indices]
        results["bm25_rank"] = range(1, len(results) + 1)

        return results[
            [
                "chunk_id",
                "document_name",
                "document_type",
                "page_number",
                "chunk_text",
                "source_url",
                "bm25_score",
                "bm25_rank",
            ]
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> pd.DataFrame:
        dense_results = self.dense_retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retrieve(query, top_k=candidate_k)

        combined = {}

        for _, row in dense_results.iterrows():
            chunk_id = row["chunk_id"]

            combined[chunk_id] = row.to_dict()
            combined[chunk_id]["bm25_score"] = 0.0
            combined[chunk_id]["bm25_rank"] = None

            combined[chunk_id]["hybrid_score"] = (
                self.dense_weight / (self.rrf_k + row["dense_rank"])
            )

        for _, row in bm25_results.iterrows():
            chunk_id = row["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = row.to_dict()
                combined[chunk_id]["dense_score"] = 0.0
                combined[chunk_id]["dense_rank"] = None
                combined[chunk_id]["hybrid_score"] = 0.0

            combined[chunk_id]["bm25_score"] = row["bm25_score"]
            combined[chunk_id]["bm25_rank"] = row["bm25_rank"]

            combined[chunk_id]["hybrid_score"] += (
                self.bm25_weight / (self.rrf_k + row["bm25_rank"])
            )

        results = pd.DataFrame(combined.values())

        results = results.sort_values(
            by="hybrid_score",
            ascending=False,
        ).head(top_k)

        results = results.reset_index(drop=True)
        results["rank"] = range(1, len(results) + 1)

        # Keep a general score column so evaluation scripts can use it easily.
        results["score"] = results["hybrid_score"]

        return results[
            [
                "rank",
                "score",
                "hybrid_score",
                "dense_score",
                "bm25_score",
                "dense_rank",
                "bm25_rank",
                "document_name",
                "document_type",
                "page_number",
                "chunk_text",
                "source_url",
                "chunk_id",
            ]
        ]


def main():
    query = input("Enter your question: ")

    retriever = HybridRetriever()
    results = retriever.retrieve(query, top_k=5, candidate_k=20)

    print("\nTop hybrid retrieved chunks:\n")

    for _, row in results.iterrows():
        print("=" * 80)
        print(f"Rank: {row['rank']}")
        print(f"Hybrid score: {row['hybrid_score']:.6f}")
        print(f"Dense score: {row['dense_score']:.4f}")
        print(f"BM25 score: {row['bm25_score']:.4f}")
        print(f"Dense rank: {row['dense_rank']}")
        print(f"BM25 rank: {row['bm25_rank']}")
        print(f"Document: {row['document_name']}")
        print(f"Type: {row['document_type']}")
        print(f"Page: {row['page_number']}")
        print(f"Source: {row['source_url']}")
        print("\nChunk:")
        print(str(row["chunk_text"])[:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()