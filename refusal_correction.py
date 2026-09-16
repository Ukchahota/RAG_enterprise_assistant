"""
Recode S3 blind labels entered as 'incorrect' by keypress error to 'refused'.

Put this file in the repo root and run:
    python recode_s3_refusals.py

Safety checks before anything changes:
  - every blind 'incorrect' row must be a question originally labelled 'refused'
  - the uncorrected sheet is backed up first
"""

import os
import sys
import shutil
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if not (ROOT / "results").exists():
    ROOT = Path.cwd()
RES = ROOT / "results"
SHEET = RES / "correctness_s3_relabel_blind.csv"
BACKUP = RES / "correctness_s3_relabel_blind_before_refusal_fix.csv"
ORIGINAL = RES / "correctness_scores.csv"
LABELS = ["correct", "partial", "incorrect", "refused"]
NOTE = ("coding correction: entered as incorrect by keypress error; "
        "annotator confirmed the answer is a refusal (rubric definition of refused)")

for p in (SHEET, ORIGINAL):
    if not p.exists():
        sys.exit(f"Missing {p}")

df = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
orig = pd.read_csv(ORIGINAL, dtype=str, keep_default_na=False)
orig["question_id"] = orig["question_id"].str.strip()
orig_label = dict(zip(orig["question_id"], orig["correctness"].str.strip().str.lower()))

blind = df["correctness"].str.strip().str.lower()
targets = df.index[blind == "incorrect"].tolist()

if not targets:
    print("No 'incorrect' labels found - nothing to recode (already done?).")
else:
    unexpected = [df.at[k, "question_id"] for k in targets
                  if orig_label.get(df.at[k, "question_id"].strip()) != "refused"]
    if unexpected:
        sys.exit(f"Stopping: these 'incorrect' rows were NOT originally refused: {unexpected}. "
                 "Paste this to Claude.")

    if not BACKUP.exists():
        shutil.copyfile(SHEET, BACKUP)
        print(f"Backup written: {BACKUP.name}")

    for k in targets:
        df.at[k, "correctness"] = "refused"
        existing = df.at[k, "notes"].strip()
        df.at[k, "notes"] = NOTE if not existing else f"{existing} | {NOTE}"

    tmp = SHEET.with_name(SHEET.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    while True:
        try:
            os.replace(tmp, SHEET)
            break
        except PermissionError:
            input("File is locked (open in Excel, or OneDrive syncing). Close it, then press Enter...")
    print(f"Recoded {len(targets)} row(s): "
          + ", ".join(sorted(df.at[k, 'question_id'] for k in targets)))

# Updated agreement with the original S3 labels
m = df[["question_id", "correctness", "notes"]].copy()
m["question_id"] = m["question_id"].str.strip()
m["blind"] = m["correctness"].str.strip().str.lower()
m["original"] = m["question_id"].map(orig_label)

n = len(m)
po = (m["original"] == m["blind"]).mean()
pe = sum((m["original"] == l).mean() * (m["blind"] == l).mean() for l in LABELS)
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

print("\n" + "=" * 72)
print("S3 BLIND RELABEL vs ORIGINAL S3 LABELS (after recode)")
print("=" * 72)
print(pd.DataFrame({
    "original": m["original"].value_counts().reindex(LABELS, fill_value=0),
    "blind": m["blind"].value_counts().reindex(LABELS, fill_value=0),
}).to_string())
print(f"\nExact agreement: {po:.3f} ({int(round(po * n))}/{n})")
print(f"Cohen's kappa (4 labels): {kappa:.3f}")
changed = m[m["original"] != m["blind"]]
print(f"Changed labels: {len(changed)}")
for _, r in changed.sort_values("question_id").iterrows():
    print(f"  {r['question_id']:<8} {r['original']:>10} -> {r['blind']:<10}")