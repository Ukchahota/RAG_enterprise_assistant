"""
Rebuilds the per-question summary from a saved claims file, applying the
current NON_CLAIM_PATTERNS / ABSTENTION_PATTERNS without re-running the NLI
model.

Why this exists: the entailment and contradiction scores in
hallucination_claims_final.csv do not change when the non-claim filter
changes - only which sentences count as claims does. Re-labelling those
sentences and recomputing the counts is arithmetic over a CSV, so a filter
change costs seconds rather than another multi-hour DeBERTa pass.

What it does:
  1. loads the claims file and re-applies is_non_claim() to every sentence
  2. any sentence now matching a non-claim or abstention pattern is
     relabelled 'nei' and so drops out of n_checkable
  3. recomputes the summary metrics via faithfulness.summarise_labels()
  4. carries question_type, difficulty, n_evidence_verified and
     gold_in_evidence over from the existing summary, so questions whose
     answers produced no sentences at all are still present as rows
  5. writes a new summary and claims file and prints a before/after diff

Usage:
    python -m src.rebuild_summary
    python -m src.rebuild_summary --in-suffix _final --out-suffix _final_v2
"""

import argparse
from pathlib import Path

import pandas as pd

from src.faithfulness import is_abstention, is_non_claim, summarise_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CARRY_COLUMNS = [
    "question_type", "difficulty", "n_evidence_verified", "gold_in_evidence",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Recompute summaries from a saved claims file."
    )
    p.add_argument("--in-suffix", default="_final",
                   help="Suffix of the input files (default: _final).")
    p.add_argument("--out-suffix", default="_final_v2",
                   help="Suffix for the rebuilt files (default: _final_v2).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing over existing output files.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    claims_in = RESULTS_DIR / f"hallucination_claims{args.in_suffix}.csv"
    summary_in = RESULTS_DIR / f"hallucination_evaluation{args.in_suffix}.csv"
    claims_out = RESULTS_DIR / f"hallucination_claims{args.out_suffix}.csv"
    summary_out = RESULTS_DIR / f"hallucination_evaluation{args.out_suffix}.csv"

    for path in (claims_in, summary_in):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    existing = [p for p in (claims_out, summary_out) if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Output file(s) already exist:\n  "
            + "\n  ".join(p.name for p in existing)
            + "\nPass --overwrite or change --out-suffix."
        )

    claims = pd.read_csv(claims_in, encoding="utf-8-sig")
    old_summary = pd.read_csv(summary_in, encoding="utf-8-sig")

    print(f"Loaded {len(claims)} claim rows, {len(old_summary)} summary rows.")

    # --- re-apply the non-claim filter ------------------------------------
    sentences = claims["sentence"].astype(str)
    claims["abstention"] = sentences.map(is_abstention)
    claims["now_non_claim"] = sentences.map(is_non_claim)

    claims["old_label"] = claims["label"]
    relabelled = claims["now_non_claim"] & (claims["label"] != "nei")
    claims.loc[claims["now_non_claim"], "label"] = "nei"

    print(f"\nRelabelled to nei: {int(relabelled.sum())} claim row(s)")
    if relabelled.any():
        print(
            claims[relabelled]
            .groupby(["system", "old_label"])
            .size().to_string()
        )
        print("\nSentences affected (deduplicated):")
        print(
            claims.loc[relabelled, "sentence"].astype(str).str[:90]
            .value_counts().to_string()
        )

    # --- recompute the summary -------------------------------------------
    grouped = {
        key: frame for key, frame in claims.groupby(["system", "question_id"])
    }

    rows = []
    for _, old in old_summary.iterrows():
        key = (old["system"], old["question_id"])
        frame = grouped.get(key)

        if frame is None:
            # No sentences were extracted from this answer at all (empty or
            # sub-threshold output). Keep the row so the denominator stays at
            # the full question set.
            metrics = summarise_labels([], any_citation=False, n_abstention=0)
        else:
            has_citation = (
                frame["cited"].notna()
                & (frame["cited"].astype(str).str.strip() != "")
            ).any()
            metrics = summarise_labels(
                frame["label"].tolist(),
                any_citation=bool(has_citation),
                n_abstention=int(frame["abstention"].sum()),
            )

        row = {"system": old["system"], "question_id": old["question_id"]}
        for col in CARRY_COLUMNS:
            row[col] = old.get(col)
        row.update(metrics)
        rows.append(row)

    summary = pd.DataFrame(rows)

    claims.drop(columns=["now_non_claim"]).to_csv(
        claims_out, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    # --- before / after ---------------------------------------------------
    print("\n" + "=" * 78)
    print("BEFORE (as verified)")
    print("=" * 78)
    print(
        old_summary.groupby("system").agg(
            questions=("question_id", "count"),
            claims=("n_checkable", "sum"),
            faithfulness=("faithfulness", "mean"),
            unsupported_rate=("unsupported_claim_rate", "mean"),
        ).round(3).to_string()
    )

    print("\n" + "=" * 78)
    print("AFTER (abstention excluded from n_checkable)")
    print("=" * 78)
    print(
        summary.groupby("system").agg(
            questions=("question_id", "count"),
            scored=("faithfulness", "count"),
            claims=("n_checkable", "sum"),
            faithfulness=("faithfulness", "mean"),
            unsupported_rate=("unsupported_claim_rate", "mean"),
        ).round(3).to_string()
    )
    print(
        "\n'questions' is every question; 'scored' is those with at least one "
        "checkable claim. The means are over 'scored' only - abstentions and "
        "empty answers are undefined, not zero, and are reported below as "
        "coverage loss instead."
    )

    print("\n--- COVERAGE / ABSTENTION ---")
    cov = summary.assign(
        abstained=lambda d: d["hallucination_risk"] == "ABSTAINED",
        no_claims=lambda d: d["hallucination_risk"] == "NO_CLAIMS",
    ).groupby("system").agg(
        questions=("question_id", "count"),
        abstained=("abstained", "sum"),
        no_claims=("no_claims", "sum"),
        scored=("faithfulness", "count"),
    )
    cov["abstention_rate"] = (cov["abstained"] / cov["questions"]).round(3)
    cov["answer_rate"] = (cov["scored"] / cov["questions"]).round(3)
    print(cov.to_string())
    print(
        "\nabstained  = deliberate refusal (template detected)\n"
        "no_claims  = empty or non-assertive output, no refusal issued - the "
        "degenerate case a claim-level verifier cannot see"
    )

    print("\n--- MICRO (claim-weighted) RATES ---")
    micro = summary.groupby("system").agg(
        n_checkable=("n_checkable", "sum"),
        n_supported=("n_supported", "sum"),
        n_supported_uncited=("n_supported_uncited", "sum"),
        n_unsupported=("n_unsupported", "sum"),
        n_contradicted=("n_contradicted", "sum"),
    )
    micro["micro_faithfulness"] = (
        (micro["n_supported"] + micro["n_supported_uncited"])
        / micro["n_checkable"].replace(0, pd.NA)
    )
    micro["micro_ungrounded"] = (
        (micro["n_unsupported"] + micro["n_contradicted"])
        / micro["n_checkable"].replace(0, pd.NA)
    )
    print(micro.round(3).to_string())
    print(
        "\nThe unsupported and contradicted columns are reported separately "
        "because the contradiction signal is low-precision on this corpus; "
        "a combined ungrounded rate inherits that unreliability."
    )

    print("\n--- RISK ---")
    print(pd.crosstab(summary["system"], summary["hallucination_risk"]).to_string())

    print(f"\nWrote {summary_out}")
    print(f"Wrote {claims_out}")


if __name__ == "__main__":
    main()