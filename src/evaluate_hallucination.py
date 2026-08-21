"""
Runs claim-level faithfulness verification over saved answer files and
produces the per-system comparison (proposal RQ1 and RQ3).

S0 has no retrieved evidence, so its claims are verified against the gold
chunks for each question — asking whether the parametric answer matches what
the policy actually says.

Set ONLY_SYSTEMS to a list of system names to re-verify a subset without
repeating the expensive full pass.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.faithfulness import verify_answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "vector_store" / "chunk_metadata.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_chunks" / "document_chunks.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

ANSWER_FILES = {
    "S0_llm_only": "answers_s0_llm_only.csv",
    "S1_basic_rag": "answers_s1_basic_rag.csv",
    "S2_modular_rag": "answers_s2_modular_rag.csv",
    "S3_corrective_rag": "answers_s3_corrective_rag.csv",
}

# Set to e.g. ["S0_llm_only"] to verify one system only. None = all available.
ONLY_SYSTEMS = ["S0_llm_only"]

# Output suffix so partial runs don't overwrite a full run.
SUFFIX = "_s0" if ONLY_SYSTEMS and len(ONLY_SYSTEMS) == 1 else ""

CLAIMS_PATH = RESULTS_DIR / f"hallucination_claims{SUFFIX}.csv"
SUMMARY_PATH = RESULTS_DIR / f"hallucination_evaluation{SUFFIX}.csv"


def load_chunk_texts() -> dict:
    """chunk_id -> chunk_text, from metadata, falling back to processed chunks."""
    for path in (METADATA_PATH, CHUNKS_PATH):
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "chunk_id" in df.columns and "chunk_text" in df.columns:
            mapping = dict(
                zip(df["chunk_id"].astype(str), df["chunk_text"].astype(str))
            )
            print(f"Loaded {len(mapping)} chunk texts from {path.name}")
            return mapping
    raise SystemExit("Could not find chunk_id/chunk_text in metadata or chunks CSV.")


def _split_ids(value) -> list[str]:
    """Split a pipe-separated ID field, treating NaN/'nan'/'' as empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return []
    return [
        x.strip() for x in text.split("|")
        if x.strip() and x.strip().lower() != "nan"
    ]


def build_evidence(row, chunk_texts) -> list[dict]:
    """Reconstruct the evidence list the model actually saw."""
    ids = _split_ids(row.get("evidence_chunk_ids"))

    if ids:
        texts = []
        raw = row.get("evidence_texts", "")
        if isinstance(raw, str) and raw.startswith("["):
            try:
                texts = json.loads(raw)
            except json.JSONDecodeError:
                texts = []
        if len(texts) != len(ids):
            texts = [chunk_texts.get(cid, "") for cid in ids]
        return [
            {"chunk_id": cid, "chunk_text": txt}
            for cid, txt in zip(ids, texts)
            if txt
        ]

    # No retrieval (S0): fall back to the gold chunks for this question.
    gold = _split_ids(row.get("gold_chunk_ids"))
    return [
        {"chunk_id": cid, "chunk_text": chunk_texts[cid]}
        for cid in gold
        if cid in chunk_texts
    ]


def main() -> None:
    chunk_texts = load_chunk_texts()

    available = {
        name: RESULTS_DIR / fname
        for name, fname in ANSWER_FILES.items()
        if (RESULTS_DIR / fname).exists()
        and (ONLY_SYSTEMS is None or name in ONLY_SYSTEMS)
    }

    if not available:
        raise SystemExit("No matching answer files found in results/.")

    print("Verifying:", ", ".join(available))

    claim_rows = []
    summary_rows = []
    empty_evidence = 0

    for system, path in available.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        print(f"\n{system}: {len(df)} answers")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=system):
            evidence = build_evidence(row, chunk_texts)
            if not evidence:
                empty_evidence += 1

            result = verify_answer(row.get("answer", ""), evidence)

            for claim in result["claims"]:
                claim_rows.append({
                    "system": system,
                    "question_id": row["question_id"],
                    "question_type": row.get("question_type", ""),
                    "label": claim["label"],
                    "sentence": claim["sentence"],
                    "cited": " ".join(str(c) for c in claim["cited"]),
                    "best_entailment": claim["best_entailment"],
                    "best_contradiction": claim["best_contradiction"],
                    "supporting_chunk": claim["supporting_chunk"],
                })

            summary_rows.append({
                "system": system,
                "question_id": row["question_id"],
                "question_type": row.get("question_type", ""),
                "difficulty": row.get("difficulty", ""),
                "n_evidence_verified": len(evidence),
                "gold_in_evidence": row.get("gold_in_evidence", None),
                "n_sentences": result["n_sentences"],
                "n_checkable": result["n_checkable"],
                "n_supported": result["n_supported"],
                "n_supported_uncited": result["n_supported_uncited"],
                "n_contradicted": result["n_contradicted"],
                "n_unsupported": result["n_unsupported"],
                "n_nei": result["n_nei"],
                "faithfulness": result["faithfulness"],
                "unsupported_claim_rate": result["unsupported_claim_rate"],
                "citation_accuracy": result["citation_accuracy"],
                "hallucination_risk": result["hallucination_risk"],
            })

    claims = pd.DataFrame(claim_rows)
    summary = pd.DataFrame(summary_rows)

    claims.to_csv(CLAIMS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    if empty_evidence:
        print(
            f"\nWARNING: {empty_evidence} answers had no evidence to verify "
            "against — their claims default to unsupported."
        )

    print("\n" + "=" * 78)
    print("CLAIM-LEVEL FAITHFULNESS")
    print("=" * 78)

    print(
        summary.groupby("system").agg(
            questions=("question_id", "count"),
            claims=("n_checkable", "sum"),
            mean_evidence=("n_evidence_verified", "mean"),
            faithfulness=("faithfulness", "mean"),
            unsupported_rate=("unsupported_claim_rate", "mean"),
            citation_accuracy=("citation_accuracy", "mean"),
        ).round(3).to_string()
    )

    print("\n--- CLAIM LABEL DISTRIBUTION ---")
    print(
        pd.crosstab(claims["system"], claims["label"], normalize="index")
        .round(3).to_string()
    )

    print("\n--- HALLUCINATION RISK ---")
    print(
        pd.crosstab(summary["system"], summary["hallucination_risk"]).to_string()
    )

    print("\n--- UNSUPPORTED RATE BY QUESTION TYPE ---")
    print(
        summary.pivot_table(
            index="question_type",
            columns="system",
            values="unsupported_claim_rate",
        ).round(3).to_string()
    )

    rag = summary[summary["gold_in_evidence"].notna()].copy()
    if not rag.empty:
        print("\n--- FAILURE SOURCE (RAG systems) ---")
        rag["retrieval_ok"] = rag["gold_in_evidence"] > 0
        rag["answer_ok"] = rag["unsupported_claim_rate"] <= 0.2
        print(
            pd.crosstab(
                [rag["system"], rag["retrieval_ok"]],
                rag["answer_ok"],
                rownames=["system", "retrieval found gold"],
                colnames=["answer grounded"],
            ).to_string()
        )
        print(
            "\nretrieval_ok=False + answer_ok=False -> retrieval failure\n"
            "retrieval_ok=True  + answer_ok=False -> generation failure"
        )

    print(f"\nSaved: {SUMMARY_PATH}")
    print(f"Saved: {CLAIMS_PATH}")


if __name__ == "__main__":
    main()