# mcnemar_tests.py
# Exact McNemar tests on paired binary outcomes.
import pandas as pd
from scipy.stats import binomtest

SUMMARY = r"results\hallucination_evaluation_final_v2.csv"
DRAFTS = r"results\hallucination_evaluation_s3_drafts.csv"


def mcnemar(name, a_ok, b_ok, a_label, b_label):
    """a_ok / b_ok: aligned boolean Series for the same items."""
    b = int((~a_ok & b_ok).sum())   # a wrong, b right
    c = int((a_ok & ~b_ok).sum())   # a right, b wrong
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else 1.0
    print(f"--- {name} ---")
    print(f"  both correct : {int((a_ok & b_ok).sum())}")
    print(f"  both wrong   : {int((~a_ok & ~b_ok).sum())}")
    print(f"  {a_label} wrong, {b_label} right : {b}")
    print(f"  {a_label} right, {b_label} wrong : {c}")
    print(f"  discordant n = {n},  exact p = {p:.4f}")
    print(f"  {'SIGNIFICANT at 0.05' if p < 0.05 else 'not significant at 0.05'}\n")


# ---- 1. Retrieval: BM25 vs reranked (hit@5), paired on question_id
r = pd.read_csv(r"results\retrieval_evaluation_final_reranked.csv")
if {"bm25_hit_at_5", "hit_at_5"}.issubset(r.columns):
    mcnemar("Retrieval: BM25 vs BM25+CrossEncoder (hit@5)",
            r["bm25_hit_at_5"].astype(bool),
            r["hit_at_5"].astype(bool),
            "BM25", "reranked")

# ---- 2. Retrieval: BM25 vs Hybrid (hit@5)
bm = pd.read_csv(r"results\retrieval_evaluation_bm25_v2_FINAL.csv")
hy = pd.read_csv(r"results\retrieval_evaluation_hybrid_v2_FINAL.csv")
key = "question_id" if "question_id" in bm.columns else bm.columns[0]
m = bm[[key, "hit_at_5"]].merge(hy[[key, "hit_at_5"]], on=key,
                                suffixes=("_bm25", "_hybrid"))
mcnemar("Retrieval: BM25 vs Hybrid (hit@5)",
        m["hit_at_5_bm25"].astype(bool),
        m["hit_at_5_hybrid"].astype(bool),
        "BM25", "hybrid")

# ---- shared: common set and current-code summaries
common = set(pd.read_csv(r"results\common_question_set.csv")["question_id"])
h = pd.read_csv(SUMMARY, encoding="utf-8-sig")
h = h[h["question_id"].isin(common)]
for col in ("n_unsupported", "n_contradicted", "n_checkable"):
    h[col] = pd.to_numeric(h[col], errors="coerce").fillna(0).astype(int)


def clean_flags(system):
    """question_id -> True if the answer had zero unsupported claims.

    Restricted to questions with at least one checkable claim: an answer that
    produced no claims carries no verifier signal and cannot be called clean.
    """
    s = h[(h["system"] == system) & (h["n_checkable"] > 0)]
    return s.set_index("question_id")["n_unsupported"].eq(0)


# ---- 3a. Do the ANSWERS S3 emits differ from S2's, on questions both answered?
# The honest comparison: refusals are excluded rather than scored as wins,
# so this asks whether S3's emitted answers are cleaner than S2's on the
# same questions. Refusal is a coverage cost and is reported separately.
s2 = clean_flags("S2_modular_rag")
s3 = clean_flags("S3_corrective_rag")
both = sorted(set(s2.index) & set(s3.index))
print(f"[3a] Questions answered by BOTH S2 and S3 (of {len(common)} common): "
      f"{len(both)}\n")
if both:
    mcnemar("Faithfulness: S2 vs S3 emitted answers (entailment only, "
            "refusals excluded)",
            s2.loc[both], s3.loc[both], "S2", "S3")

# ---- 3b. Did withholding remove bad answers? Draft vs final, within S3.
# Pairs each refused question's draft against the counterfactual that it had
# been emitted. This measures the layer's discrimination, not its accuracy:
# the trigger fires ON the unsupported count, so the association is partly
# definitional and must be reported as such.
d = pd.read_csv(DRAFTS, encoding="utf-8-sig")
for col in ("n_unsupported", "n_checkable"):
    d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
d = d[d["question_id"].isin(common) & (d["n_checkable"] > 0)]
drafts_clean = d.set_index("question_id")["n_unsupported"].eq(0)

accepted = set(s3.index)
refused = sorted(set(drafts_clean.index) - accepted)
print(f"[3b] Refused drafts in the common set with checkable claims: "
      f"{len(refused)}")
if refused:
    n_clean = int(drafts_clean.loc[refused].sum())
    print(f"     Of these, drafts with zero unsupported claims: {n_clean} "
          f"({n_clean / len(refused):.1%})")
    print(f"     Drafts with at least one unsupported claim: "
          f"{len(refused) - n_clean}")
    p = binomtest(n_clean, len(refused), 0.5).pvalue
    print(f"     Exact binomial vs 50/50: p = {p:.4g}")
    print("     Interpretation: a low clean-draft proportion means the layer "
          "withheld\n     answers that were in fact ungrounded. Partly "
          "definitional - the trigger\n     fires on this same quantity.\n")

# ---- 3c. Coverage cost
cov = h[h["system"] == "S3_corrective_rag"]
n_abst = int((cov["hallucination_risk"] == "ABSTAINED").sum())
print(f"[3c] Coverage: S3 abstained on {n_abst} of {len(cov)} common-set "
      f"questions ({n_abst / len(cov):.1%})")
print("     Faithfulness gains above are conditional on answering; this is "
      "the cost.")