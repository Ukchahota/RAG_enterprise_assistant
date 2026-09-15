"""
Sample claims for manual validation of the NLI detector (proposal 7.4).

Produces a CSV with a blank human_label column for hand-annotation.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "results" / "hallucination_claims.csv"
METADATA_PATH = PROJECT_ROOT / "vector_store" / "chunk_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "detector_validation_sample.csv"

PER_LABEL = 15
SEED = 42


def main() -> None:
    claims = pd.read_csv(CLAIMS_PATH, encoding="utf-8-sig")
    meta = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    chunk_text = dict(
        zip(meta["chunk_id"].astype(str), meta["chunk_text"].astype(str))
    )

    # Stratify across predicted labels so every class is represented.
    frames = []
    for label, group in claims.groupby("label"):
        n = min(PER_LABEL, len(group))
        frames.append(group.sample(n=n, random_state=SEED))

    sample = pd.concat(frames).sample(frac=1, random_state=SEED).reset_index(drop=True)

    sample["supporting_chunk_text"] = (
        sample["supporting_chunk"].astype(str).map(chunk_text).fillna("")
    )

    out = sample[[
        "question_id", "question_type", "sentence", "cited",
        "supporting_chunk", "supporting_chunk_text",
        "best_entailment", "best_contradiction", "label",
    ]].copy()

    out = out.rename(columns={"label": "predicted_label"})
    out["human_label"] = ""
    out["notes"] = ""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Sampled {len(out)} claims for manual validation.")
    print(out["predicted_label"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nFill in human_label using exactly one of:")
    print("  supported | supported_uncited | contradicted | unsupported | nei")


if __name__ == "__main__":
    main()