"""
Can a runtime signal substitute for gold-based failure attribution?

Gold-based attribution needs gold chunk labels, which a deployed system does
not have. This tests whether the cross-encoder relevance score, available at
query time, can stand in for them:

    gold_attr     Retrieval or Generation, from gold_in_evidence (>= 1.0)
    runtime_attr  Retrieval or Generation, from the best CE score vs a cutoff

Three corrections to the earlier version:

1. Labels. It read draft_n_unsupported / draft_n_contradicted from the S3
   answers file - computed inline during the August generation run by a
   verifier with a different sentence splitter, so attribution was tested
   against labels no other result uses. This reads the rebuilt _final_v2
   summary and defaults to S2, an uncorrected system whose emitted answers
   were actually scored.

2. Retrieval completeness. .fillna(False).astype(bool) mapped 0.5 to True.
   Complete retrieval requires >= 1.0.

3. Threshold. The earlier version hardcoded a CE cutoff of 0 while the
   retrieval pipeline uses -1.0. A single arbitrary cutoff makes a low kappa
   dismissible as a tuning artefact, so this sweeps the cutoff. If kappa
   stays near zero everywhere, the negative result is a property of the
   signal, not of one bad threshold.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

RESULTS_DIR = Path("results")

GOLD_COMPLETE_MIN = 1.0
PIPELINE_CE_THRESHOLD = -1.0

ANSWER_FILES = {
    "S1_basic_rag": "answers_s1_basic_rag.csv",
    "S2_modular_rag": "answers_s2_modular_rag.csv",
    "S3_corrective_rag": "answers_s3_corrective_rag.csv",
}


def best_ce_score(value) -> float:
    """Highest cross-encoder score in a pipe-separated evidence_scores field."""
    if pd.isna(value):
        return float("nan")
    scores = []
    for token in str(value).split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            scores.append(float(token))
        except ValueError:
            continue
    return max(scores) if scores else float("nan")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Runtime vs gold attribution.")
    p.add_argument("--system", default="S2_modular_rag", choices=list(ANSWER_FILES))
    p.add_argument("--summary", default="_final_v2")
    p.add_argument("--unsupported-only", action="store_true",
                   help="Count only n_unsupported, excluding the low-precision "
                        "contradiction signal.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    summary_path = RESULTS_DIR / f"hallucination_evaluation{args.summary}.csv"
    answers_path = RESULTS_DIR / ANSWER_FILES[args.system]
    for path in (summary_path, answers_path):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary = summary[summary["system"] == args.system].copy()
    answers = pd.read_csv(answers_path, encoding="utf-8-sig")

    scores = answers[["question_id", "evidence_scores"]].copy()
    scores["best_ce"] = scores["evidence_scores"].map(best_ce_score)

    d = summary.merge(scores[["question_id", "best_ce"]], on="question_id",
                      how="left")

    for col in ("n_unsupported", "n_contradicted", "n_checkable"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

    d["gold_frac"] = pd.to_numeric(d["gold_in_evidence"], errors="coerce")
    d["retrieval_complete"] = d["gold_frac"] >= GOLD_COMPLETE_MIN
    d["ungrounded"] = (
        d["n_unsupported"] if args.unsupported_only
        else d["n_unsupported"] + d["n_contradicted"]
    )

    print(f"System: {args.system}"
          f"{'  (unsupported only)' if args.unsupported_only else ''}")
    print(f"Questions: {len(d)}   with a CE score: {int(d['best_ce'].notna().sum())}")
    print(f"CE score range: {d['best_ce'].min():.3f} to {d['best_ce'].max():.3f}")

    scoreable = d[(d["n_checkable"] > 0) & d["best_ce"].notna()].copy()
    failures = scoreable[scoreable["ungrounded"] > 0].copy()
    print(f"Excluded (no checkable claims or no CE score): {len(d) - len(scoreable)}")
    print(f"Failures available for attribution: {len(failures)}")

    if failures.empty:
        raise SystemExit("No failures to attribute.")

    failures["gold_attr"] = [
        "Generation" if c else "Retrieval" for c in failures["retrieval_complete"]
    ]
    print("\nGold attribution:")
    print(failures["gold_attr"].value_counts().to_string())

    print("\n" + "=" * 78)
    print("AGREEMENT ACROSS CE CUTOFFS")
    print("=" * 78)
    print(f"{'cutoff':>8}  {'agree':>6}  {'kappa':>7}  {'pred R':>7}  {'pred G':>7}")

    lo = float(failures["best_ce"].min())
    hi = float(failures["best_ce"].max())
    span = hi - lo
    cutoffs = sorted({round(lo + span * i / 12, 3) for i in range(13)}
                     | {PIPELINE_CE_THRESHOLD, 0.0})

    best = None
    for cutoff in cutoffs:
        runtime = ["Retrieval" if ce < cutoff else "Generation"
                   for ce in failures["best_ce"]]
        agree = (failures["gold_attr"].to_numpy()
                 == pd.Series(runtime).to_numpy()).mean()
        if len(set(runtime)) < 2 or failures["gold_attr"].nunique() < 2:
            kappa = float("nan")
        else:
            kappa = cohen_kappa_score(failures["gold_attr"], runtime)
        mark = ""
        if abs(cutoff - PIPELINE_CE_THRESHOLD) < 1e-9:
            mark = "  <- pipeline threshold"
        elif cutoff == 0.0:
            mark = "  <- previous hardcoded cutoff"
        print(f"{cutoff:>8.3f}  {agree:>6.3f}  {kappa:>7.3f}  "
              f"{runtime.count('Retrieval'):>7}  {runtime.count('Generation'):>7}{mark}")
        if best is None or (kappa == kappa and kappa > best[1]):
            best = (cutoff, kappa, agree, runtime)

    majority = failures["gold_attr"].value_counts(normalize=True).max()
    print(f"\nMajority-class baseline accuracy: {majority:.3f}")
    print("Kappa corrects for chance agreement, so a kappa near zero means the")
    print("runtime signal carries no information about which layer failed, even")
    print("where raw agreement looks superficially high.")

    if best is not None:
        cutoff, kappa, agree, runtime = best
        print(f"\nBest cutoff by kappa: {cutoff:.3f} "
              f"(kappa={kappa:.3f}, agreement={agree:.3f})")
        print("\nConfusion matrix at that cutoff:")
        print(pd.crosstab(failures["gold_attr"],
                          pd.Series(runtime, index=failures.index),
                          rownames=["gold"], colnames=["runtime"]).to_string())

    print("\n" + "=" * 78)
    print("CE SCORE DISTRIBUTION BY GOLD LAYER")
    print("=" * 78)
    print(failures.groupby("gold_attr")["best_ce"]
          .agg(["count", "mean", "std", "min", "median", "max"]).round(3).to_string())
    print("\nIf the two distributions overlap, no cutoff can separate them, and")
    print("the negative result follows from the signal rather than the choice of")
    print("threshold. The mechanism: a cross-encoder scores topical relevance to")
    print("the query, which is not the same property as whether the retrieved")
    print("passage is sufficient to answer it.")

    out_path = RESULTS_DIR / f"attribution_agreement_{args.system}.csv"
    failures[["question_id", "question_type", "gold_frac", "retrieval_complete",
              "ungrounded", "best_ce", "gold_attr"]].to_csv(
        out_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()