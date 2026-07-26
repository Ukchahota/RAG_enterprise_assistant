from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

FILES = {
    "V1_FAISS_Dense": RESULTS_DIR / "retrieval_evaluation_current.csv",
    "V2_BM25": RESULTS_DIR / "retrieval_evaluation_bm25.csv",
    "V3_Hybrid": RESULTS_DIR / "retrieval_evaluation_hybrid.csv",
    "V4_Query_Rewritten_Hybrid": RESULTS_DIR / "retrieval_evaluation_query_rewritten.csv",
}

OUTPUT_PATH = RESULTS_DIR / "retriever_comparison_summary.csv"


def summarise_results(version_name: str, file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing result file: {file_path}")

    df = pd.read_csv(file_path)

    return {
        "version": version_name,
        "total_questions": len(df),
        "top_1_hit_rate": df["top_1_hit"].mean(),
        "hit_at_3": df["hit_at_3"].mean(),
        "hit_at_5": df["hit_at_5"].mean(),
        "precision_at_5": df["precision_at_5"].mean(),
        "mrr_at_5": df["mrr_at_5"].mean(),
        "average_retrieval_time_seconds": df["retrieval_time_seconds"].mean(),
    }


def main() -> None:
    rows = []

    for version_name, file_path in FILES.items():
        rows.append(summarise_results(version_name, file_path))

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\nRetriever Comparison Summary")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()