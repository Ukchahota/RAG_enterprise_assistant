from pathlib import Path

import pandas as pd

from src.hybrid_retriever import HybridRetriever
from src.query_rewritten_retriever import (
    QueryRewrittenRetriever,
    detect_policy_intent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CorrectiveRetriever:
    """
    Corrective RAG retriever controller.

    Stage 1:
    - Retrieve using Hybrid FAISS + BM25.

    Stage 2:
    - Assess whether retrieval is good enough.

    Stage 3:
    - If retrieval is weak, apply corrective retrieval:
      query rewriting + metadata-aware policy fallback.
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        print("Loading initial hybrid retriever...")
        self.initial_retriever = HybridRetriever(
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )

        print("Loading corrective query-rewritten retriever...")
        self.corrective_retriever = QueryRewrittenRetriever(
            original_weight=0.4,
            rewritten_weight=0.6,
            policy_weight=1.0,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            policy_metadata_boost=0.05,
        )

    def policy_document_present(
        self,
        results: pd.DataFrame,
        policy_type: str,
    ) -> bool:
        """
        Check whether the requested policy document appears in retrieved results.
        """
        if results.empty:
            return False

        searchable_text = (
            results.get("document_name", "").astype(str)
            + " "
            + results.get("document_type", "").astype(str)
            + " "
            + results.get("source_url", "").astype(str)
        ).str.lower()

        if policy_type == "payment":
            return searchable_text.str.contains("payment", na=False).any() and (
                searchable_text.str.contains("policy", na=False).any()
                or searchable_text.str.contains("feespayment", na=False).any()
            )

        if policy_type == "conduct":
            return searchable_text.str.contains("conduct", na=False).any()

        return False

    def assess_retrieval_quality(
        self,
        query: str,
        results: pd.DataFrame,
    ) -> tuple[bool, list[str]]:
        """
        Decide whether initial retrieval is good enough.

        Returns:
        - quality_passed: True/False
        - reasons: list of reasons for correction
        """
        reasons = []

        if results.empty:
            reasons.append("no_results")
            return False, reasons

        policy_type = detect_policy_intent(query)

        if policy_type is not None:
            if not self.policy_document_present(results, policy_type):
                reasons.append(f"requested_{policy_type}_policy_not_found")

        if "score" in results.columns:
            top_score = float(results.iloc[0]["score"])

            if top_score <= 0:
                reasons.append("top_score_zero_or_negative")

        quality_passed = len(reasons) == 0

        return quality_passed, reasons

    def add_corrective_metadata(
        self,
        results: pd.DataFrame,
        correction_applied: bool,
        quality_passed: bool,
        correction_reasons: list[str],
        retrieval_stage: str,
    ) -> pd.DataFrame:
        """
        Add metadata columns so evaluation can explain what happened.
        """
        results = results.copy()

        if "final_score" not in results.columns:
            results["final_score"] = results.get("score", 0.0)

        if "query_rewrite_score" not in results.columns:
            results["query_rewrite_score"] = results.get("score", 0.0)

        if "metadata_boost" not in results.columns:
            results["metadata_boost"] = 0.0

        if "hybrid_score" not in results.columns:
            results["hybrid_score"] = results.get("score", 0.0)

        if "original_query" not in results.columns:
            results["original_query"] = ""

        if "rewritten_query" not in results.columns:
            results["rewritten_query"] = ""

        if "rewrite_applied" not in results.columns:
            results["rewrite_applied"] = False

        if "policy_fallback_rank" not in results.columns:
            results["policy_fallback_rank"] = None

        if "policy_fallback_score" not in results.columns:
            results["policy_fallback_score"] = 0.0

        results["correction_applied"] = correction_applied
        results["quality_passed"] = quality_passed
        results["correction_reasons"] = ", ".join(correction_reasons)
        results["retrieval_stage"] = retrieval_stage

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
    ) -> pd.DataFrame:
        """
        Run corrective retrieval.
        """
        initial_results = self.initial_retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        quality_passed, correction_reasons = self.assess_retrieval_quality(
            query=query,
            results=initial_results,
        )

        if quality_passed:
            final_results = self.add_corrective_metadata(
                results=initial_results,
                correction_applied=False,
                quality_passed=True,
                correction_reasons=[],
                retrieval_stage="initial_hybrid",
            )

            return final_results

        corrected_results = self.corrective_retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        final_results = self.add_corrective_metadata(
            results=corrected_results,
            correction_applied=True,
            quality_passed=False,
            correction_reasons=correction_reasons,
            retrieval_stage="corrective_query_rewrite",
        )

        return final_results


def main():
    query = input("Enter your question: ")

    retriever = CorrectiveRetriever()
    results = retriever.retrieve(query, top_k=5, candidate_k=50)

    print("\nTop corrective retrieved chunks:\n")

    if results.empty:
        print("No results found.")
        return

    print(f"Correction applied: {results.iloc[0]['correction_applied']}")
    print(f"Quality passed: {results.iloc[0]['quality_passed']}")
    print(f"Correction reasons: {results.iloc[0]['correction_reasons']}")
    print(f"Retrieval stage: {results.iloc[0]['retrieval_stage']}")

    if "rewritten_query" in results.columns:
        print("\nRewritten query:")
        print(results.iloc[0]["rewritten_query"])

    print()

    for _, row in results.iterrows():
        print("=" * 80)
        print(f"Rank: {row['rank']}")
        print(f"Score: {row['score']:.6f}")
        print(f"Document: {row['document_name']}")
        print(f"Type: {row['document_type']}")
        print(f"Page: {row['page_number']}")
        print(f"Source: {row['source_url']}")

        if "policy_fallback_rank" in row:
            print(f"Policy fallback rank: {row['policy_fallback_rank']}")

        print("\nChunk:")
        print(str(row["chunk_text"])[:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()