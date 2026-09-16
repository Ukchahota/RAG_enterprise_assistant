"""
Runs claim-level faithfulness verification over saved answer files and
produces the per-system comparison (proposal RQ1 and RQ3).

S0 has no retrieved evidence, so its claims are verified against the gold
chunks for each question - asking whether the parametric answer matches what
the policy actually says. S1-S3 are verified against the evidence actually
retrieved for that question (k=5). The S0 comparison is therefore
conservative: S0 is scored against 1-2 premises where the RAG systems get 5,
so S0's measured faithfulness is if anything understated relative to them.

Usage
-----
  # single system, full file
  python -m src.evaluate_hallucination --system S2_modular_rag --suffix _s2_rerun

  # smoke test on a 3-row file
  python -m src.evaluate_hallucination --system S2_modular_rag \
      --answers _smoke_s2.csv --suffix _smoke

  # split run, to survive an out-of-memory kill
  python -m src.evaluate_hallucination --system S2_modular_rag \
      --answers _s2_part1.csv --suffix _s2a
  python -m src.evaluate_hallucination --system S2_modular_rag \
      --answers _s2_part2.csv --suffix _s2b

  # continue an interrupted run
  python -m src.evaluate_hallucination --system S2_modular_rag \
      --suffix _s2_rerun --resume

  # report only, no verification (recompute the printout from saved CSVs)
  python -m src.evaluate_hallucination --suffix _s2_rerun --report-only

Rows are flushed to disk every --flush-every questions, so a crash loses at
most that many; --resume picks up from whatever is already on disk.
"""

import argparse
import json
import sys
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

# --- analysis thresholds -----------------------------------------------------
# gold_in_evidence is a FRACTION of the gold chunks present in the retrieved
# evidence, not a boolean. Complete retrieval therefore means 1.0, not "> 0":
# a question with 1 of 2 gold chunks retrieved (0.5) is a PARTIAL retrieval and
# must not be counted as a retrieval success. The previous "> 0" test silently
# folded partial retrievals into the success cell of the failure-source table.
GOLD_COMPLETE_MIN = 1.0

# An answer counts as grounded when at most this fraction of its checkable
# claims are unsupported or contradicted. NOTE: this is deliberately stricter
# than the corrective layer's trigger (0.40). The two thresholds answer
# different questions - "is this answer grounded enough to call a success?"
# versus "is this answer bad enough to be worth withholding?" - and the gap
# between them is reported below so the asymmetry is visible rather than
# buried.
GROUNDED_MAX_UNSUPPORTED_RATE = 0.2
CORRECTIVE_TRIGGER_RATE = 0.4

# Column order is pinned so that appended batches, and separately-run parts,
# concatenate cleanly and stay schema-compatible with the existing
# hallucination_evaluation_s1_s3.csv / _s0_rerun.csv outputs.
SUMMARY_COLUMNS = [
    "system", "question_id", "question_type", "difficulty",
    "n_evidence_verified", "gold_in_evidence", "n_sentences", "n_checkable",
    "n_supported", "n_supported_uncited", "n_contradicted", "n_unsupported",
    "n_nei", "faithfulness", "unsupported_claim_rate", "citation_accuracy",
    "hallucination_risk",
]

CLAIM_COLUMNS = [
    "system", "question_id", "question_type", "label", "sentence", "cited",
    "best_entailment", "best_contradiction", "supporting_chunk",
]


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


class IncrementalWriter:
    """Buffers rows and appends them to a CSV, so a crash loses only the buffer."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        self.buffer: list[dict] = []
        self.written = 0

    def add(self, row: dict) -> None:
        self.buffer.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer, columns=self.columns)
        is_new = not self.path.exists()
        df.to_csv(
            self.path,
            mode="w" if is_new else "a",
            header=is_new,
            index=False,
            # utf-8-sig only on creation, or the BOM is re-inserted mid-file.
            encoding="utf-8-sig" if is_new else "utf-8",
        )
        self.written += len(self.buffer)
        self.buffer.clear()


def load_completed(summary_path: Path, claims_path: Path) -> set:
    """(system, question_id) pairs already verified, for --resume.

    The summary file is treated as the source of truth. If the claims file
    contains rows for questions the summary does not have (a crash between the
    two flushes), those orphan claim rows are dropped so the two files stay
    consistent.
    """
    if not summary_path.exists():
        return set()

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    done = {
        (str(s), str(q))
        for s, q in zip(summary["system"], summary["question_id"])
    }

    if claims_path.exists():
        claims = pd.read_csv(claims_path, encoding="utf-8-sig")
        mask = [
            (str(s), str(q)) in done
            for s, q in zip(claims["system"], claims["question_id"])
        ]
        if not all(mask):
            dropped = len(claims) - sum(mask)
            claims[mask].to_csv(claims_path, index=False, encoding="utf-8-sig")
            print(f"Dropped {dropped} orphan claim rows to match the summary file.")

    print(f"Resuming: {len(done)} question(s) already verified.")
    return done


def verify(args, summary_path: Path, claims_path: Path) -> None:
    chunk_texts = load_chunk_texts()

    wanted = args.system or list(ANSWER_FILES)
    unknown = [s for s in wanted if s not in ANSWER_FILES]
    if unknown:
        raise SystemExit(f"Unknown system(s): {', '.join(unknown)}")

    if args.answers and len(wanted) != 1:
        raise SystemExit("--answers can only be used with a single --system.")

    available = {}
    for name in wanted:
        fname = args.answers if args.answers else ANSWER_FILES[name]
        path = RESULTS_DIR / fname
        if path.exists():
            available[name] = path
        else:
            print(f"Skipping {name}: {path.name} not found.")

    if not available:
        raise SystemExit("No matching answer files found in results/.")

    done = load_completed(summary_path, claims_path) if args.resume else set()

    print("Verifying:", ", ".join(f"{k} <- {v.name}" for k, v in available.items()))

    summary_writer = IncrementalWriter(summary_path, SUMMARY_COLUMNS)
    claims_writer = IncrementalWriter(claims_path, CLAIM_COLUMNS)

    empty_evidence = 0
    zero_claim_ids: list[str] = []

    try:
        for system, path in available.items():
            df = pd.read_csv(path, encoding="utf-8-sig")
            if args.limit:
                df = df.head(args.limit)
            print(f"\n{system}: {len(df)} answers")

            since_flush = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=system):
                qid = str(row["question_id"])
                if (system, qid) in done:
                    continue

                evidence = build_evidence(row, chunk_texts)
                if not evidence:
                    empty_evidence += 1

                result = verify_answer(row.get("answer", ""), evidence)

                if result["n_checkable"] == 0:
                    zero_claim_ids.append(f"{system}:{qid}")

                for claim in result["claims"]:
                    claims_writer.add({
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

                summary_writer.add({
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

                since_flush += 1
                if since_flush >= args.flush_every:
                    claims_writer.flush()   # claims first...
                    summary_writer.flush()  # ...summary last, so it stays authoritative
                    since_flush = 0
    finally:
        # Always persist whatever has been computed, including on Ctrl-C.
        claims_writer.flush()
        summary_writer.flush()
        print(f"\nWrote {summary_writer.written} summary row(s) this run.")

    if empty_evidence:
        print(
            f"\nWARNING: {empty_evidence} answer(s) had no evidence to verify "
            "against - their claims default to unsupported. For an S1-S3 run "
            "this indicates an evidence-parsing problem, not a model failure."
        )

    if zero_claim_ids:
        print(
            f"\nNOTE: {len(zero_claim_ids)} answer(s) yielded zero checkable "
            "claims (empty or non-assertive output):"
        )
        print("  " + ", ".join(zero_claim_ids))


def print_report(summary_path: Path, claims_path: Path) -> None:
    if not summary_path.exists():
        raise SystemExit(f"No summary file at {summary_path}")

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    claims = (
        pd.read_csv(claims_path, encoding="utf-8-sig")
        if claims_path.exists() else pd.DataFrame(columns=CLAIM_COLUMNS)
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
    print(
        "\nThe table above is a MACRO mean: each question contributes equally, "
        "whatever its claim count."
    )

    # Micro rates, so the macro figures can be compared against a
    # claim-weighted denominator. faithfulness and unsupported_claim_rate are
    # complements over n_checkable, and n_nei is excluded from n_checkable, so
    # recomputing either from the label distribution below will NOT match the
    # macro table unless the denominators are matched first.
    micro = summary.groupby("system").agg(
        n_checkable=("n_checkable", "sum"),
        n_supported=("n_supported", "sum"),
        n_supported_uncited=("n_supported_uncited", "sum"),
        n_unsupported=("n_unsupported", "sum"),
        n_contradicted=("n_contradicted", "sum"),
        n_nei=("n_nei", "sum"),
    )
    micro["micro_faithfulness"] = (
        (micro["n_supported"] + micro["n_supported_uncited"])
        / micro["n_checkable"].replace(0, pd.NA)
    )
    micro["micro_ungrounded_rate"] = (
        (micro["n_unsupported"] + micro["n_contradicted"])
        / micro["n_checkable"].replace(0, pd.NA)
    )
    print("\n--- MICRO (claim-weighted) RATES ---")
    print(micro.round(3).to_string())

    zero = summary[summary["n_checkable"] == 0]
    if not zero.empty:
        print(
            f"\n--- ZERO-CLAIM ANSWERS ({len(zero)}) ---\n"
            "These produced no checkable claims, so no verifier signal exists "
            "for them. A claim-level verifier is structurally blind to this "
            "failure mode: zero claims means a 0.0 unsupported rate, which "
            f"sits below the corrective trigger ({CORRECTIVE_TRIGGER_RATE}) "
            "and is accepted."
        )
        print(
            zero.groupby("system")["question_id"]
            .apply(lambda s: ", ".join(map(str, s))).to_string()
        )
        nonzero = summary[summary["n_checkable"] > 0]
        print("\nMacro means excluding zero-claim answers:")
        print(
            nonzero.groupby("system").agg(
                questions=("question_id", "count"),
                faithfulness=("faithfulness", "mean"),
                unsupported_rate=("unsupported_claim_rate", "mean"),
            ).round(3).to_string()
        )

    if not claims.empty:
        print("\n--- CLAIM LABEL DISTRIBUTION (share of all labelled claims) ---")
        print(
            pd.crosstab(claims["system"], claims["label"], normalize="index")
            .round(3).to_string()
        )
        print(
            "\nNOTE: this denominator includes nei claims, which are excluded "
            "from n_checkable above. The two tables are not directly comparable."
        )

    print("\n--- HALLUCINATION RISK ---")
    print(pd.crosstab(summary["system"], summary["hallucination_risk"]).to_string())

    print("\n--- UNSUPPORTED RATE BY QUESTION TYPE ---")
    print(
        summary.pivot_table(
            index="question_type",
            columns="system",
            values="unsupported_claim_rate",
        ).round(3).to_string()
    )

    # --- failure source ------------------------------------------------------
    rag = summary.copy()
    rag["gold_frac"] = pd.to_numeric(rag["gold_in_evidence"], errors="coerce")
    rag = rag[rag["gold_frac"].notna()].copy()

    if not rag.empty:
        print("\n--- RETRIEVAL COMPLETENESS ---")
        buckets = pd.cut(
            rag["gold_frac"],
            bins=[-0.001, 0.0, 0.999, 1.0],
            labels=["none (0.0)", "partial (0<f<1)", "complete (1.0)"],
        )
        print(pd.crosstab(rag["system"], buckets).to_string())
        print(
            "\nThe 'partial' column is what a '> 0' test would have scored as "
            "retrieval success. It is counted as retrieval FAILURE below, "
            f"since complete retrieval requires gold_in_evidence >= "
            f"{GOLD_COMPLETE_MIN}."
        )

        print("\n--- FAILURE SOURCE (RAG systems) ---")
        rag["retrieval_ok"] = rag["gold_frac"] >= GOLD_COMPLETE_MIN
        rag["answer_ok"] = (
            rag["unsupported_claim_rate"] <= GROUNDED_MAX_UNSUPPORTED_RATE
        )
        print(
            pd.crosstab(
                [rag["system"], rag["retrieval_ok"]],
                rag["answer_ok"],
                rownames=["system", "retrieval complete"],
                colnames=["answer grounded"],
            ).to_string()
        )
        print(
            "\nretrieval_ok=False + answer_ok=False -> retrieval-layer failure\n"
            "retrieval_ok=True  + answer_ok=False -> generation-layer failure\n"
            f"'grounded' = ungrounded claim rate <= {GROUNDED_MAX_UNSUPPORTED_RATE}; "
            f"the corrective layer triggers at {CORRECTIVE_TRIGGER_RATE}, so "
            "answers between the two are counted as failures here but were not "
            "withheld at runtime.\n"
            "This is an association between layer state and outcome, not a "
            "causal attribution: no intervention is applied."
        )

    print(f"\nSummary: {summary_path}")
    print(f"Claims:  {claims_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Claim-level faithfulness verification over saved answers."
    )
    p.add_argument(
        "--system", action="append", metavar="NAME",
        help=f"System to verify (repeatable). One of: {', '.join(ANSWER_FILES)}. "
             "Default: all available.",
    )
    p.add_argument(
        "--answers", metavar="FILENAME",
        help="Override the answers filename in results/ (single --system only). "
             "Use for smoke tests and split runs.",
    )
    p.add_argument(
        "--suffix", default="",
        help="Output suffix, e.g. _s2_rerun -> "
             "results/hallucination_evaluation_s2_rerun.csv",
    )
    p.add_argument(
        "--flush-every", type=int, default=5, metavar="N",
        help="Append rows to disk every N questions (default: 5).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip questions already present in the output summary.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing output files for this suffix and start fresh.",
    )
    p.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Verify only the first N answers per system (quick check).",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip verification and reprint the report from saved CSVs.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    summary_path = RESULTS_DIR / f"hallucination_evaluation{args.suffix}.csv"
    claims_path = RESULTS_DIR / f"hallucination_claims{args.suffix}.csv"

    if args.report_only:
        print_report(summary_path, claims_path)
        return

    exists = [p for p in (summary_path, claims_path) if p.exists()]
    if exists and not (args.resume or args.overwrite):
        print(
            "Refusing to run: output file(s) already exist and would be "
            "appended to, duplicating rows:\n  "
            + "\n  ".join(p.name for p in exists)
            + "\n\nPass --resume to continue an interrupted run, --overwrite to "
              "start fresh, or change --suffix.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.overwrite:
        for p in exists:
            p.unlink()
            print(f"Deleted {p.name}")

    verify(args, summary_path, claims_path)
    print_report(summary_path, claims_path)


if __name__ == "__main__":
    main()