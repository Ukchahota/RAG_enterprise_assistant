"""
Scores the NLI detector against manual labels (proposal 7.4).

Reports per-class precision, recall, F1, macro/weighted averages,
false positive and false negative rates, and a confusion matrix.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "results" / "detector_validation_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "detector_validation_metrics.csv"

LABELS = ["supported", "supported_uncited", "contradicted", "unsupported", "nei"]
# Binary view: which labels mean "this claim is a hallucination"?
POSITIVE = {"unsupported", "contradicted"}


def main() -> None:
    df = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    labelled = df[df["human_label"].isin(LABELS)].copy()

    print(f"Labelled: {len(labelled)} of {len(df)}")
    if len(labelled) < 20:
        raise SystemExit("Label at least 20 claims before scoring.")

    y_true = labelled["human_label"]
    y_pred = labelled["predicted_label"]

    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    print(classification_report(y_true, y_pred, zero_division=0))

    print("=" * 70)
    print("CONFUSION MATRIX (rows = human, cols = detector)")
    print("=" * 70)
    present = [x for x in LABELS if x in set(y_true) | set(y_pred)]
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=present),
        index=present, columns=present,
    )
    print(cm.to_string())

    print("\n" + "=" * 70)
    print("BINARY VIEW: hallucination detection")
    print("=" * 70)
    t = y_true.isin(POSITIVE)
    p = y_pred.isin(POSITIVE)
    tp = int((t & p).sum()); fp = int((~t & p).sum())
    fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"  TP {tp} | FP {fp} | FN {fn} | TN {tn}")
    print(f"  Precision:           {precision:.3f}")
    print(f"  Recall:              {recall:.3f}")
    print(f"  F1:                  {f1:.3f}")
    print(f"  False positive rate: {fp / (fp + tn) if fp + tn else 0:.3f}")
    print(f"  False negative rate: {fn / (fn + tp) if fn + tp else 0:.3f}")
    print(f"  Cohen's kappa:       {cohen_kappa_score(y_true, y_pred):.3f}")

    print("\n" + "=" * 70)
    print("DISAGREEMENTS")
    print("=" * 70)
    bad = labelled[y_true != y_pred]
    for _, r in bad.head(10).iterrows():
        print(f"\n  human={r['human_label']} detector={r['predicted_label']} "
              f"(ent {r['best_entailment']:.2f} con {r['best_contradiction']:.2f})")
        print(f"  {str(r['sentence'])[:140]}")

    cm.to_csv(OUTPUT_PATH, encoding="utf-8-sig")
    print(f"\nSaved confusion matrix: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()