"""
Failure-layer attribution.

For each question, classifies the outcome by which pipeline layer the failure
is attributable to, using gold retrieval labels as ground truth.

Layers
------
    Clean        no ungrounded claims and no misattributed citations
    Attribution  fully grounded, but claims are cited wrongly or not at all
    Retrieval    ungrounded claims present AND retrieval was incomplete
    Generation   ungrounded claims present DESPITE complete retrieval
    Parametric   S0 only: ungrounded with no retrieval stage to blame
    Abstained    the corrective layer withheld the answer
    NoClaims     no checkable claims were extracted (empty or structured
                 output) - the verifier has no signal, so no layer can be
                 assigned

Two corrections to the earlier version of this analysis:

1. Input. It previously read results/hallucination_evaluation.csv, which
   predates the current verifier: it holds an S0 run that was given no
   premises at all, a different sentence splitter, and no S1 or S3 at all.
   S3 was patched in from the draft_* columns of the answers file, which were
   likewise computed by the old code. This version reads the rebuilt
   _final_v2 summary, so all systems come from one code version.

2. Retrieval completeness. gold_in_evidence is the FRACTION of gold chunks
   present in the retrieved evidence. The previous
   .fillna(False).astype(bool) coercion mapped 0.5 to True, so a question
   with one of two gold chunks retrieved counted as a retrieval success and
   any resulting failure was attributed to generation. Complete retrieval
   requires >= 1.0. The sensitivity section below reports how many questions
   this reclassifies.

Note on interpretation: this is an association between layer state and
outcome, not a causal attribution. No intervention is applied, and context
length and evidence position are uncontrolled confounds.

Usage:
    python analyse_failure_layers.py
    python analyse_failure_layers.py --summary _final_v2
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

# Complete retrieval means every gold chunk is in the evidence window.
GOLD_COMPLETE_MIN = 1.0

# Systems that perform retrieval, in reporting order.
RAG_SYSTEMS = ["S1_basic_rag", "S2_modular_rag", "S3_corrective_rag"]

# Systems whose answers are uncorrected, and so whose failures are the
# relevant subject for layer attribution. S3's scored answers are
# post-correction: its failures have been withheld rather than emitted, so
# its layer distribution describes what the corrective layer let through, not
# where the pipeline fails.
UNCORRECTED = ["S1_basic_rag", "S2_modular_rag"]


def classify(row, gold_complete: bool) -> str:
    """Assign a failure layer to one question."""
    if row["hallucination_risk"] == "ABSTAINED":
        return "Abstained"
    if row["n_checkable"] == 0:
        return "NoClaims"

    ungrounded = row["n_unsupported"] + row["n_contradicted"]

    if row["system"] == "S0_llm_only":
        return "Parametric" if ungrounded > 0 else "Clean"

    if ungrounded == 0:
        return "Attribution" if row["n_supported_uncited"] > 0 else "Clean"

    return "Generation" if gold_complete else "Retrieval"


def add_layers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    gold = pd.to_numeric(out["gold_in_evidence"], errors="coerce")
    complete = gold >= threshold
    out["retrieval_complete"] = complete
    out["failure_layer"] = [
        classify(row, bool(c)) for (_, row), c in zip(out.iterrows(), complete)
    ]
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Failure-layer attribution.")
    p.add_argument("--summary", default="_final_v2",
                   help="Suffix of the summary file to read (default: _final_v2).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    path = RESULTS_DIR / f"hallucination_evaluation{args.summary}.csv"
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")

    raw = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("n_unsupported", "n_contradicted", "n_supported_uncited",
                "n_checkable"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)

    print(f"Loaded {len(raw)} rows from {path.name}")
    print(f"Systems: {', '.join(sorted(raw['system'].unique()))}\n")

    df = add_layers(raw, GOLD_COMPLETE_MIN)

    order = ["Clean", "Attribution", "Retrieval", "Generation", "Parametric",
             "Abstained", "NoClaims"]
    present = [c for c in order if c in set(df["failure_layer"])]

    print("=" * 78)
    print("FAILURE LAYER BY SYSTEM (counts)")
    print("=" * 78)
    counts = pd.crosstab(df["system"], df["failure_layer"])
    print(counts.reindex(columns=present, fill_value=0).to_string())

    print("\n" + "=" * 78)
    print("FAILURE LAYER BY SYSTEM (row-normalised)")
    print("=" * 78)
    print(
        pd.crosstab(df["system"], df["failure_layer"], normalize="index")
        .reindex(columns=present, fill_value=0).round(3).to_string()
    )

    # --- the headline ratio, on uncorrected systems only ------------------
    print("\n" + "=" * 78)
    print("RETRIEVAL vs GENERATION (uncorrected systems)")
    print("=" * 78)
    for system in UNCORRECTED:
        sub = df[df["system"] == system]
        r = int((sub["failure_layer"] == "Retrieval").sum())
        g = int((sub["failure_layer"] == "Generation").sum())
        ratio = f"{g / r:.2f}" if r else "n/a"
        print(f"{system:<20} retrieval={r:<4} generation={g:<4} G:R={ratio}")

    pooled = df[df["system"].isin(UNCORRECTED)]
    r = int((pooled["failure_layer"] == "Retrieval").sum())
    g = int((pooled["failure_layer"] == "Generation").sum())
    print(f"{'POOLED':<20} retrieval={r:<4} generation={g:<4} "
          f"G:R={(g / r) if r else float('nan'):.2f}")
    print(
        "\nS3 is excluded from this comparison because its scored answers are\n"
        "post-correction: failures it detected were withheld rather than\n"
        "emitted, so counting them would understate the pipeline's failures."
    )

    # --- by question type -------------------------------------------------
    print("\n" + "=" * 78)
    print("RETRIEVAL vs GENERATION BY QUESTION TYPE (uncorrected systems)")
    print("=" * 78)
    fails = pooled[pooled["failure_layer"].isin(["Retrieval", "Generation"])]
    if not fails.empty:
        tab = pd.crosstab(fails["question_type"], fails["failure_layer"])
        for col in ("Retrieval", "Generation"):
            if col not in tab:
                tab[col] = 0
        tab["G:R"] = (tab["Generation"] / tab["Retrieval"].replace(0, pd.NA)).round(2)
        print(tab[["Retrieval", "Generation", "G:R"]].to_string())
        print(
            "\nThis breakdown matters more than the pooled ratio: if the two\n"
            "layers dominate different question types, a correction mechanism\n"
            "that only re-retrieves is adequate for some question classes and\n"
            "not others, which is a narrower and more defensible claim than a\n"
            "single overall ratio."
        )

    # --- what the corrective layer withheld -------------------------------
    s3 = df[df["system"] == "S3_corrective_rag"]
    abst = s3[s3["failure_layer"] == "Abstained"]
    if not abst.empty:
        print("\n" + "=" * 78)
        print("ABSTENTIONS BY RETRIEVAL STATE (S3)")
        print("=" * 78)
        print(abst["retrieval_complete"].value_counts().rename(
            {True: "retrieval complete", False: "retrieval incomplete"}
        ).to_string())
        print(
            "\nAbstentions on COMPLETE retrieval are cases re-retrieval could\n"
            "not have fixed: the evidence was already present and the draft\n"
            "still failed verification. Abstentions on INCOMPLETE retrieval\n"
            "are the subset a CRAG-style re-retrieval step might address."
        )
        print("\nBy question type:")
        print(abst["question_type"].value_counts().to_string())

    # --- threshold sensitivity -------------------------------------------
    print("\n" + "=" * 78)
    print("SENSITIVITY: gold_in_evidence >= 1.0 vs > 0")
    print("=" * 78)
    loose = add_layers(raw, 1e-9)  # any gold chunk counts as complete
    for system in RAG_SYSTEMS:
        a = df[df["system"] == system]["failure_layer"]
        b = loose[loose["system"] == system]["failure_layer"]
        print(
            f"{system:<20} "
            f">=1.0: R={int((a == 'Retrieval').sum()):<3} "
            f"G={int((a == 'Generation').sum()):<3}   "
            f">0: R={int((b == 'Retrieval').sum()):<3} "
            f"G={int((b == 'Generation').sum()):<3}"
        )

    gold = pd.to_numeric(raw["gold_in_evidence"], errors="coerce")
    partial = raw[(gold > 0) & (gold < 1.0)]
    print(f"\nPartial retrievals (0 < f < 1): {len(partial)} question(s)")
    if not partial.empty:
        print(partial.groupby("system")["question_id"].count().to_string())
        print(
            "\nThese are the questions the previous boolean coercion "
            "misclassified as\nretrieval successes."
        )

    out_path = RESULTS_DIR / f"failure_layers{args.summary}.csv"
    df[["system", "question_id", "question_type", "difficulty",
        "gold_in_evidence", "retrieval_complete", "n_checkable",
        "n_supported", "n_supported_uncited", "n_unsupported",
        "n_contradicted", "hallucination_risk", "failure_layer"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()