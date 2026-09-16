# score_detector.py
"""
Scores the NLI verifier against human labels on the stratified validation
sample (n=63).

Three corrections:

1. Input. It read detector_validation_human.csv, which is byte-identical to
   detector_validation_human_pass1.csv - so the reconciled second-pass labels
   were never used. This reads _final and reports pass1-vs-pass2 agreement
   separately as an intra-annotator reliability figure.

2. The supported / supported_uncited distinction. The blind sample shown to
   the annotator omitted the citation field, so the annotator could not judge
   whether a claim was correctly cited - only whether the evidence entailed
   it. Comparing a 5-class human label against a 5-class model prediction
   therefore scores the annotator on a distinction they could not see. The
   primary analysis collapses both into 'entailed'; the citation-dependent
   split is reported separately as model output only, NOT validated.

3. Recall and F1 are not interpretable here. The sample was stratified by
   PREDICTED label, so for each predicted class we have a random sample of
   that class and can estimate precision. Recall requires sampling
   proportional to true classes, which this design does not provide.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score

KEY = r"results\detector_validation_key.csv"
FINAL = r"results\detector_validation_human_final.csv"
PASS1 = r"results\detector_validation_human_pass1.csv"
PASS2 = r"results\detector_validation_human_pass2.csv"

FULL_LABELS = ["supported", "supported_uncited", "contradicted",
               "unsupported", "nei"]
COLLAPSED = ["entailed", "contradicted", "unsupported", "nei"]

COLLAPSE = {
    "supported": "entailed",
    "supported_uncited": "entailed",
    "contradicted": "contradicted",
    "unsupported": "unsupported",
    "nei": "nei",
}


def label_column(df, name):
    for candidate in ("human_label", "label", "verdict", "human"):
        if candidate in df.columns:
            return candidate
    raise SystemExit(f"No label column found in {name}: {list(df.columns)}")


key = pd.read_csv(KEY)
final = pd.read_csv(FINAL)
lab = label_column(final, "final")

df = key[["row_id", "question_type", "predicted_label", "cited",
          "best_entailment", "best_contradiction"]].merge(
    final[["row_id", lab]], on="row_id")
df = df.rename(columns={lab: "human_label"})
df = df[df["human_label"].notna() & (df["human_label"].astype(str).str.strip() != "")]

print(f"Scored {len(df)}/{len(key)} claims from {FINAL.split(chr(92))[-1]}\n")

# ---- intra-annotator reliability -------------------------------------------
try:
    p1 = pd.read_csv(PASS1)
    p2 = pd.read_csv(PASS2)
    c1, c2 = label_column(p1, "pass1"), label_column(p2, "pass2")
    pair = p1[["row_id", c1]].merge(p2[["row_id", c2]], on="row_id",
                                    suffixes=("_1", "_2"))
    a, b = pair.columns[1], pair.columns[2]
    pair = pair[pair[a].notna() & pair[b].notna()]
    same = (pair[a] == pair[b])
    print("--- INTRA-ANNOTATOR RELIABILITY (pass 1 vs pass 2) ---")
    print(f"  items relabelled  : {len(pair)}")
    print(f"  identical labels  : {int(same.sum())} ({same.mean():.3f})")
    if pair[a].nunique() > 1 and pair[b].nunique() > 1:
        print(f"  Cohen's kappa     : "
              f"{cohen_kappa_score(pair[a], pair[b]):.3f}")
    changed = pair[~same]
    if not changed.empty:
        print(f"\n  Changed between passes ({len(changed)}):")
        print("  " + changed.to_string(index=False).replace("\n", "\n  "))
    print("\n  This is the same annotator relabelling, so it bounds label")
    print("  stability - it is not inter-annotator agreement, which this")
    print("  project has no second annotator to provide.\n")
except FileNotFoundError as e:
    print(f"(pass1/pass2 comparison skipped: {e})\n")

# ---- primary analysis: collapsed classes -----------------------------------
df["pred_c"] = df["predicted_label"].map(COLLAPSE)
df["human_c"] = df["human_label"].map(COLLAPSE)

unmapped = df[df["human_c"].isna()]["human_label"].unique()
if len(unmapped):
    print(f"WARNING: unmapped human labels: {list(unmapped)}\n")

d = df[df["pred_c"].notna() & df["human_c"].notna()]

print("=" * 74)
print("PRIMARY: PER-CLASS PRECISION (collapsed - citation not human-judged)")
print("=" * 74)
print(f"  Raw agreement: {(d['pred_c'] == d['human_c']).mean():.3f}")
print("  (stratified sample - NOT population accuracy)\n")

rows = []
for cls in COLLAPSED:
    n_pred = int((d["pred_c"] == cls).sum())
    if n_pred == 0:
        rows.append({"class": cls, "n_predicted": 0, "precision": None})
        continue
    correct = int(((d["pred_c"] == cls) & (d["human_c"] == cls)).sum())
    rows.append({"class": cls, "n_predicted": n_pred, "n_correct": correct,
                 "precision": round(correct / n_pred, 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  Precision only. Recall and F1 are omitted because the sample was")
print("  stratified by predicted label, which does not support estimating")
print("  recall. Small n_predicted means a wide interval on that precision.")

print("\n--- CONFUSION MATRIX (rows = human, cols = model) ---")
print(pd.DataFrame(
    confusion_matrix(d["human_c"], d["pred_c"], labels=COLLAPSED),
    index=COLLAPSED, columns=COLLAPSED).to_string())

# ---- what the model predicted, by human verdict ----------------------------
print("\n--- WHERE THE MODEL'S ENTAILMENT SCORE SAT ---")
print(d.groupby(["pred_c", "human_c"])[["best_entailment", "best_contradiction"]]
      .agg(["count", "mean"]).round(3).to_string())

# ---- the uncollapsed view, for completeness --------------------------------
print("\n" + "=" * 74)
print("SECONDARY: FULL 5-CLASS VIEW (supported vs supported_uncited is")
print("model output only and is NOT validated against human judgement)")
print("=" * 74)
print(pd.crosstab(df["human_label"], df["predicted_label"]).to_string())

print("\n--- DISAGREEMENTS (collapsed) ---")
dis = d[d["pred_c"] != d["human_c"]]
print(dis[["row_id", "question_type", "predicted_label", "human_label",
           "best_entailment", "best_contradiction"]].to_string(index=False))

out = r"results\detector_validation_scored.csv"
d.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\nWrote {out}")