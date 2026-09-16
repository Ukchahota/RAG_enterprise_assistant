"""
Counterfactual sweep of the corrective layer's refusal triggers.

The layer withheld an answer when EITHER
  (a) any draft claim was labelled contradicted, OR
  (b) the fraction of unsupported draft claims exceeded 0.40.

Both triggers are recomputed here from the draft claim labels, so the
consequences of a different threshold - or of dropping the contradiction
trigger entirely - can be read off without regenerating anything.

Reported per configuration:
  refusals          how many answers would be withheld
  abstention_rate   refusals / questions
  bad_caught        withheld drafts that DID have >=1 unsupported claim
  clean_withheld    withheld drafts with NO unsupported claims (candidate
                    false refusals)
  bad_released      emitted drafts that DID have >=1 unsupported claim

'clean' means the entailment check found no problem, not that the answer is
verified correct: a claim can be factually wrong yet entailed by a retrieved
chunk. These are candidates, not confirmed false refusals - and where
retrieval was incomplete, absence of contradiction is not evidence of
correctness.
"""

import pandas as pd

DRAFTS = r"results\hallucination_evaluation_s3_drafts.csv"
ANSWERS = r"results\answers_s3_corrective_rag.csv"
COMMON = r"results\common_question_set.csv"

d = pd.read_csv(DRAFTS, encoding="utf-8-sig")
for col in ("n_checkable", "n_unsupported", "n_contradicted"):
    d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

a = pd.read_csv(ANSWERS, encoding="utf-8-sig")
common = set(pd.read_csv(COMMON)["question_id"])

d = d.merge(a[["question_id", "gold_in_evidence"]], on="question_id",
            how="left", suffixes=("", "_ans"))
d["gold_frac"] = pd.to_numeric(d["gold_in_evidence"], errors="coerce")

# Only the refused questions were re-verified, so the sweep describes the
# refused population. Accepted answers are already known to pass both
# triggers at the deployed settings.
scorable = d[d["n_checkable"] > 0].copy()
scorable["unsup_rate"] = scorable["n_unsupported"] / scorable["n_checkable"]
scorable["has_contra"] = scorable["n_contradicted"] > 0
scorable["is_bad"] = scorable["n_unsupported"] > 0

n_accepted = int((a["corrective_action"] == "accepted").sum())

print(f"Refused drafts with checkable claims: {len(scorable)}")
print(f"Accepted answers (not re-verified): {n_accepted}")
print(f"Zero-claim refused drafts excluded: "
      f"{int((d['n_checkable'] == 0).sum())}\n")

rows = []
for use_contra in (True, False):
    for thresh in (0.2, 0.3, 0.4, 0.5, 0.6, 1.01):
        refuse = (scorable["unsup_rate"] > thresh)
        if use_contra:
            refuse = refuse | scorable["has_contra"]
        held = scorable[refuse]
        released = scorable[~refuse]
        total = len(scorable) + n_accepted
        rows.append({
            "contra_trigger": "on" if use_contra else "OFF",
            "unsup_thresh": thresh if thresh <= 1 else "disabled",
            "refusals": len(held),
            "abstention_rate": round(len(held) / total, 3),
            "bad_caught": int(held["is_bad"].sum()),
            "clean_withheld": int((~held["is_bad"]).sum()),
            "bad_released": int(released["is_bad"].sum()),
        })

out = pd.DataFrame(rows)
print("=" * 78)
print("TRIGGER CONFIGURATION SWEEP")
print("=" * 78)
print(out.to_string(index=False))
print("\nDeployed configuration: contra_trigger=on, unsup_thresh=0.4")
print("Rates are over all 102 questions (refused + accepted).")

# --- precision of withholding ----------------------------------------------
print("\n" + "=" * 78)
print("WITHHOLDING PRECISION (of drafts withheld, share genuinely ungrounded)")
print("=" * 78)
out["withhold_precision"] = (
    out["bad_caught"] / out["refusals"].replace(0, pd.NA)
).astype("Float64").round(3)
print(out[["contra_trigger", "unsup_thresh", "refusals",
           "withhold_precision", "clean_withheld"]].to_string(index=False))

# --- retrieval state of the clean-withheld cases ---------------------------
print("\n" + "=" * 78)
print("CANDIDATE FALSE REFUSALS BY RETRIEVAL STATE (deployed config)")
print("=" * 78)
deployed = scorable[(scorable["unsup_rate"] > 0.4) | scorable["has_contra"]]
clean = deployed[~deployed["is_bad"]]
print(f"Clean drafts withheld: {len(clean)}")
if not clean.empty:
    complete = clean["gold_frac"] >= 1.0
    print(f"  retrieval complete   : {int(complete.sum())}  "
          f"(clearest false refusals - evidence was present)")
    print(f"  retrieval incomplete : {int((~complete).sum())}  "
          f"(unclear - absence of contradiction is not evidence of "
          f"correctness)")
    print("\n" + clean[["question_id", "question_type", "gold_frac",
                        "n_checkable", "n_contradicted"]]
          .to_string(index=False))

out.to_csv(r"results\trigger_sweep.csv", index=False, encoding="utf-8-sig")
print("\nWrote results\\trigger_sweep.csv")