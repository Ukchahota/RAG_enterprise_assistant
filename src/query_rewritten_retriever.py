import re
from pathlib import Path

import pandas as pd

from src.hybrid_retriever import HybridRetriever
from src.bm25_retriever import tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def unique_terms(terms: list[str]) -> list[str]:
    seen = set()
    output = []

    for term in terms:
        term = clean_spaces(term.lower())

        if term and term not in seen:
            seen.add(term)
            output.append(term)

    return output


def detect_policy_intent(query: str) -> str | None:
    """
    Detect whether the query explicitly asks for a named policy document.
    """
    query_lower = query.lower()

    if "payment policy" in query_lower or (
        "payment" in query_lower and "policy" in query_lower
    ):
        return "payment"

    if "student conduct policy" in query_lower or (
        "conduct" in query_lower and "policy" in query_lower
    ):
        return "conduct"

    return None


def rewrite_query(query: str) -> str:
    """
    Rule-based query rewriting / query expansion for university policy RAG.

    This is intentionally deterministic so retrieval experiments are repeatable.
    """
    query_lower = query.lower()

    expansion_terms = []

    if any(
        term in query_lower
        for term in [
            "fee",
            "fees",
            "tuition",
            "payment",
            "pay",
            "paid",
            "non-payment",
            "debt",
        ]
    ):
        expansion_terms.extend(
            [
                "payment policy",
                "programme fees",
                "tuition fees",
                "non-payment",
                "outstanding balance",
                "student debt",
                "sanctions",
                "payment deadline",
                "instalment",
                "restricted access",
                "library services",
                "mws account",
                "canvas",
                "registration",
                "award",
                "degree",
                "fees office",
            ]
        )

    if any(
        term in query_lower
        for term in ["policy", "according to", "procedure", "regulation"]
    ):
        expansion_terms.extend(
            [
                "official policy",
                "formal policy",
                "procedures",
                "regulations",
                "student programme fees",
                "accommodation fees",
                "fines and charges",
            ]
        )

    expansion_terms = unique_terms(expansion_terms)

    if not expansion_terms:
        return clean_spaces(query)

    rewritten_query = query + " " + " ".join(expansion_terms)

    return clean_spaces(rewritten_query)


class QueryRewrittenRetriever:
    """
    Selective corrective retriever.

    Normal questions:
    - Use normal Hybrid FAISS + BM25 retrieval.

    Explicit policy questions:
    - Rewrite the query with policy terms.
    - Retrieve using original query and rewritten query.
    - Add a policy-document fallback when a named policy is requested.
    """

    def __init__(
        self,
        original_weight: float = 0.4,
        rewritten_weight: float = 0.6,
        policy_weight: float = 1.0,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        policy_metadata_boost: float = 0.05,
    ):
        self.original_weight = original_weight
        self.rewritten_weight = rewritten_weight
        self.policy_weight = policy_weight
        self.rrf_k = rrf_k
        self.policy_metadata_boost = policy_metadata_boost

        self.hybrid_retriever = HybridRetriever(
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )

    def _ensure_output_columns(
        self,
        results: pd.DataFrame,
        query: str,
        rewritten_query: str,
        rewrite_applied: bool,
    ) -> pd.DataFrame:
        """
        Make sure all expected output columns exist.
        This keeps evaluation scripts consistent.
        """
        results = results.copy()

        default_columns = {
            "final_score": results.get("score", 0.0),
            "query_rewrite_score": results.get("score", 0.0),
            "metadata_boost": 0.0,
            "hybrid_score": results.get("hybrid_score", results.get("score", 0.0)),
            "dense_score": results.get("dense_score", 0.0),
            "bm25_score": results.get("bm25_score", 0.0),
            "dense_rank": results.get("dense_rank", None),
            "bm25_rank": results.get("bm25_rank", None),
            "original_rank": results.get("rank", None),
            "rewritten_rank": None,
            "policy_fallback_rank": None,
            "policy_fallback_score": 0.0,
        }

        for column, default_value in default_columns.items():
            if column not in results.columns:
                results[column] = default_value

        results["score"] = results["final_score"]
        results["original_query"] = query
        results["rewritten_query"] = rewritten_query
        results["rewrite_applied"] = rewrite_applied

        return results

    def policy_document_candidates(
        self,
        query: str,
        rewritten_query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Retrieve chunks from a specific policy document when the query explicitly
        asks for that policy.

        This is a corrective retrieval fallback.
        """
        policy_type = detect_policy_intent(query)

        if policy_type is None:
            return pd.DataFrame()

        metadata = self.hybrid_retriever.bm25_retriever.metadata.copy()

        def get_col(column_name: str) -> pd.Series:
            if column_name in metadata.columns:
                return metadata[column_name].astype(str)
            return pd.Series([""] * len(metadata), index=metadata.index)

        searchable_metadata = (
            get_col("document_name")
            + " "
            + get_col("document_type")
            + " "
            + get_col("file_name")
            + " "
            + get_col("source_url")
        ).str.lower()

        if policy_type == "payment":
            mask = (
                searchable_metadata.str.contains("payment", na=False)
                & searchable_metadata.str.contains("policy", na=False)
            ) | searchable_metadata.str.contains("feespayment", na=False)

        elif policy_type == "conduct":
            mask = searchable_metadata.str.contains("conduct", na=False)

        else:
            return pd.DataFrame()

        policy_chunks = metadata[mask].copy()

        if policy_chunks.empty:
            return pd.DataFrame()

        query_tokens = set(tokenize(rewritten_query))

        fallback_scores = []

        for chunk_text in policy_chunks["chunk_text"].astype(str).tolist():
            chunk_tokens = set(tokenize(chunk_text))
            overlap_score = len(query_tokens.intersection(chunk_tokens))

            chunk_lower = chunk_text.lower()

            phrase_bonus = 0

            if policy_type == "payment":
                important_phrases = [
                    "non-payment",
                    "programme fees",
                    "outstanding balance",
                    "sanctions",
                    "registration",
                    "award",
                    "tuition fees",
                    "payment plan",
                    "instalment",
                ]

                for phrase in important_phrases:
                    if phrase in chunk_lower:
                        phrase_bonus += 3

            elif policy_type == "conduct":
                important_phrases = [
                    "student conduct",
                    "misconduct",
                    "disciplinary",
                    "sanctions",
                    "procedure",
                    "investigation",
                ]

                for phrase in important_phrases:
                    if phrase in chunk_lower:
                        phrase_bonus += 3

            fallback_scores.append(overlap_score + phrase_bonus)

        policy_chunks["policy_fallback_score"] = fallback_scores

        policy_chunks = policy_chunks.sort_values(
            by="policy_fallback_score",
            ascending=False,
        ).head(top_k)

        policy_chunks["rank"] = range(1, len(policy_chunks) + 1)

        return policy_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
    ) -> pd.DataFrame:
        query = clean_spaces(query)

        policy_intent = detect_policy_intent(query)

        # For normal questions, do not rewrite.
        # This avoids query drift and keeps performance close to the hybrid baseline.
        if policy_intent is None:
            results = self.hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
            )

            results = self._ensure_output_columns(
                results=results,
                query=query,
                rewritten_query=query,
                rewrite_applied=False,
            )

            return results[
                [
                    "rank",
                    "score",
                    "final_score",
                    "query_rewrite_score",
                    "metadata_boost",
                    "hybrid_score",
                    "dense_score",
                    "bm25_score",
                    "dense_rank",
                    "bm25_rank",
                    "original_rank",
                    "rewritten_rank",
                    "policy_fallback_rank",
                    "policy_fallback_score",
                    "document_name",
                    "document_type",
                    "page_number",
                    "chunk_text",
                    "source_url",
                    "chunk_id",
                    "original_query",
                    "rewritten_query",
                    "rewrite_applied",
                ]
            ]

        # For explicit policy questions, apply corrective query rewriting.
        rewritten_query = rewrite_query(query)

        original_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        rewritten_results = self.hybrid_retriever.retrieve(
            query=rewritten_query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        policy_results = self.policy_document_candidates(
            query=query,
            rewritten_query=rewritten_query,
            top_k=10,
        )

        combined = {}

        for _, row in original_results.iterrows():
            chunk_id = row["chunk_id"]

            combined[chunk_id] = row.to_dict()
            combined[chunk_id]["original_rank"] = row["rank"]
            combined[chunk_id]["rewritten_rank"] = None
            combined[chunk_id]["policy_fallback_rank"] = None
            combined[chunk_id]["policy_fallback_score"] = 0.0

            combined[chunk_id]["query_rewrite_score"] = (
                self.original_weight / (self.rrf_k + row["rank"])
            )

            combined[chunk_id]["metadata_boost"] = 0.0

        for _, row in rewritten_results.iterrows():
            chunk_id = row["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = row.to_dict()
                combined[chunk_id]["original_rank"] = None
                combined[chunk_id]["policy_fallback_rank"] = None
                combined[chunk_id]["policy_fallback_score"] = 0.0
                combined[chunk_id]["query_rewrite_score"] = 0.0
                combined[chunk_id]["metadata_boost"] = 0.0

            combined[chunk_id]["rewritten_rank"] = row["rank"]

            combined[chunk_id]["query_rewrite_score"] += (
                self.rewritten_weight / (self.rrf_k + row["rank"])
            )

        for _, row in policy_results.iterrows():
            chunk_id = row["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = row.to_dict()
                combined[chunk_id]["dense_score"] = 0.0
                combined[chunk_id]["bm25_score"] = 0.0
                combined[chunk_id]["dense_rank"] = None
                combined[chunk_id]["bm25_rank"] = None
                combined[chunk_id]["hybrid_score"] = 0.0
                combined[chunk_id]["original_rank"] = None
                combined[chunk_id]["rewritten_rank"] = None
                combined[chunk_id]["query_rewrite_score"] = 0.0

            combined[chunk_id]["policy_fallback_rank"] = row["rank"]
            combined[chunk_id]["policy_fallback_score"] = row[
                "policy_fallback_score"
            ]

            combined[chunk_id]["query_rewrite_score"] += (
                self.policy_weight / (self.rrf_k + row["rank"])
            )

            # Strong but selective boost because the user explicitly requested
            # a named policy document.
            combined[chunk_id]["metadata_boost"] = self.policy_metadata_boost

        results = pd.DataFrame(combined.values())

        if results.empty:
            return pd.DataFrame()

        results["metadata_boost"] = results["metadata_boost"].fillna(0.0)

        results["final_score"] = (
            results["query_rewrite_score"] + results["metadata_boost"]
        )

        results = results.sort_values(
            by="final_score",
            ascending=False,
        ).head(top_k)

        results = results.reset_index(drop=True)
        results["rank"] = range(1, len(results) + 1)
        results["score"] = results["final_score"]
        results["original_query"] = query
        results["rewritten_query"] = rewritten_query
        results["rewrite_applied"] = True

        results = self._ensure_output_columns(
            results=results,
            query=query,
            rewritten_query=rewritten_query,
            rewrite_applied=True,
        )

        return results[
            [
                "rank",
                "score",
                "final_score",
                "query_rewrite_score",
                "metadata_boost",
                "hybrid_score",
                "dense_score",
                "bm25_score",
                "dense_rank",
                "bm25_rank",
                "original_rank",
                "rewritten_rank",
                "policy_fallback_rank",
                "policy_fallback_score",
                "document_name",
                "document_type",
                "page_number",
                "chunk_text",
                "source_url",
                "chunk_id",
                "original_query",
                "rewritten_query",
                "rewrite_applied",
            ]
        ]


def main():
    query = input("Enter your question: ")

    retriever = QueryRewrittenRetriever()
    results = retriever.retrieve(query, top_k=5, candidate_k=50)

    print("\nOriginal query:")
    print(query)

    if results.empty:
        print("\nNo results found.")
        return

    print("\nRewritten query:")
    print(results.iloc[0]["rewritten_query"])

    print("\nTop query-rewritten retrieved chunks:\n")

    for _, row in results.iterrows():
        print("=" * 80)
        print(f"Rank: {row['rank']}")
        print(f"Final score: {row['final_score']:.6f}")
        print(f"Query rewrite score: {row['query_rewrite_score']:.6f}")
        print(f"Metadata boost: {row['metadata_boost']:.6f}")
        print(f"Policy fallback rank: {row['policy_fallback_rank']}")
        print(f"Policy fallback score: {row['policy_fallback_score']}")
        print(f"Original rank: {row['original_rank']}")
        print(f"Rewritten rank: {row['rewritten_rank']}")
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