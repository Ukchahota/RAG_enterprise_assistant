"""
Blind re-labelling of S3 correctness under the committed rubric.

Put this file in the repo root and run:
    python relabel_s3_blind.py

First run : verifies the S3 answer text, builds results/correctness_s3_relabel_blind.csv,
            then pauses so you can commit it.
Later runs: resumes labelling from the first blank row.
When done : prints agreement with the original S3 labels (never modifies the original file).
"""

import os
import sys
import shutil
import textwrap
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

SOURCE = RES / "correctness_sample.csv"          # S3 answer text used in the original labelling
ORIGINAL = RES / "correctness_scores.csv"        # original S3 labels
ANSWER_FILES = [RES / "answers_s3_corrective_rag_clean.csv", RES / "answers_s3_corrective_rag.csv"]
SHEET = RES / "correctness_s3_relabel_blind.csv"

KEYS = {"c": "correct", "p": "partial", "i": "incorrect", "r": "refused"}
LABELS = ["correct", "partial", "incorrect", "refused"]
RUBRIC = ("Rubric: correct = all key facts, no contradicting specifics | "
          "partial = some key facts (scores zero) | "
          "incorrect = wrong specifics or generic answer lacking the Liverpool-specific fact | "
          "refused = declines / says it cannot know")


def norm(s):
    return " ".join(str(s).split())


def verify_source(src):
    ids = set(src["question_id"])
    orig_ids = set(pd.read_csv(ORIGINAL, dtype=str, keep_default_na=False)["question_id"].str.strip())
    if ids != orig_ids or len(src) != 29:
        sys.exit(f"{SOURCE.name} does not contain exactly the 29 questions in {ORIGINAL.name}. Stopping.")

    if "corrective_action" in src.columns:
        print("corrective_action counts in source:")
        print(src["corrective_action"].value_counts().to_string())

    any_full_match = False
    for f in ANSWER_FILES:
        if not f.exists():
            continue
        ans = pd.read_csv(f, dtype=str, keep_default_na=False)
        ans["question_id"] = ans["question_id"].str.strip()
        ans = ans.drop_duplicates("question_id").set_index("question_id")
        missing = [q for q in ids if q not in ans.index]
        differ = [q for q in ids if q in ans.index
                  and norm(ans.at[q, "answer"]) != norm(src.loc[src["question_id"] == q, "answer"].iloc[0])]
        print(f"\n{f.name}: {29 - len(missing) - len(differ)}/29 identical answers"
              f"{', missing: ' + str(sorted(missing)) if missing else ''}"
              f"{', different: ' + str(sorted(differ)) if differ else ''}")
        if not missing and not differ:
            any_full_match = True
    if not any_full_match:
        sys.exit("\nThe answer text in correctness_sample.csv does not fully match an S3 answers file. "
                 "Paste this output to Claude before labelling.")


def build_sheet():
    for p in (SOURCE, ORIGINAL):
        if not p.exists():
            sys.exit(f"Missing {p}")
    src = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    src["question_id"] = src["question_id"].str.strip()
    verify_source(src)

    sheet = src[["question_id", "question_type", "question", "expected_answer", "answer"]].rename(
        columns={"answer": "s3_answer"}).copy()
    sheet = sheet.sample(frac=1, random_state=11).reset_index(drop=True)
    sheet["correctness"] = ""
    sheet["notes"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8")
    print(f"\nBuilt {SHEET} (29 rows, corrective_action and original labels removed, shuffled with seed 11).")
    print("\nCommit it before labelling so the audit trail shows it was created first:")
    print(f'  git add "{SHEET.relative_to(ROOT)}" relabel_s3_blind.py')
    print('  git commit -m "S3 blind relabel sheet, before labelling"')
    input("\nPress Enter to start labelling (or Ctrl+C to stop and commit first)...")


if not SHEET.exists():
    build_sheet()

df = pd.read_csv(SHEET, dtype=str, keep_default_na=False)


def save():
    tmp = SHEET.with_name(SHEET.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    while True:
        try:
            os.replace(tmp, SHEET)
            return
        except PermissionError:
            input("File is locked (open in Excel, or OneDrive syncing). "
                  "Close it, then press Enter to retry...")


def show(label, text):
    width = max(60, shutil.get_terminal_size().columns - 4)
    lines = []
    for para in str(text).splitlines():
        lines.append(textwrap.fill(para, width, initial_indent="  ", subsequent_indent="  ")
                     if para.strip() else "")
    print(label)
    print("\n".join(lines))
    print()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def label_loop():
    history, skipped, force = [], set(), None
    while True:
        todo = [k for k in df.index if not df.at[k, "correctness"].strip()]
        if not todo:
            return True
        if force is not None:
            k, force = force, None
        else:
            pending = [k for k in todo if k not in skipped] or todo
            if len(pending) == len(todo):
                skipped.clear()
            k = pending[0]

        row = df.loc[k]
        clear()
        print(f"[{len(df) - len(todo)}/{len(df)} labelled]  {row['question_id']}  ({row['question_type']})\n")
        show("QUESTION:", row["question"])
        show("EXPECTED ANSWER:", row["expected_answer"])
        show("S3 ANSWER:", row["s3_answer"])
        print(textwrap.fill(RUBRIC, max(60, shutil.get_terminal_size().columns - 4)) + "\n")
        print("For partial or incorrect, note which key fact is missing or wrong.\n")

        choice = input("[c]orrect [p]artial [i]ncorrect [r]efused | "
                       "[s]kip [u]ndo [q]uit > ").strip().lower()
        if choice == "q":
            return False
        if choice == "s":
            skipped.add(k)
        elif choice == "u":
            if history:
                last, old_note = history.pop()
                df.at[last, "correctness"] = ""
                df.at[last, "notes"] = old_note
                save()
                force = last
            else:
                input("Nothing to undo - press Enter")
        elif choice in KEYS:
            note = input("Note: ").strip()
            history.append((k, df.at[k, "notes"]))
            df.at[k, "correctness"] = KEYS[choice]
            df.at[k, "notes"] = note
            save()
        else:
            input("Unrecognised key - press Enter to try again")


def cohen_kappa(a, b):
    n = len(a)
    po = (a == b).mean()
    pe = sum((a == l).sum() / n * (b == l).sum() / n for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def agreement():
    orig = pd.read_csv(ORIGINAL, dtype=str, keep_default_na=False)
    orig["question_id"] = orig["question_id"].str.strip()
    m = orig[["question_id", "correctness"]].rename(columns={"correctness": "original"}).merge(
        df[["question_id", "correctness", "notes"]].rename(columns={"correctness": "blind"}),
        on="question_id")
    m["original"] = m["original"].str.strip().str.lower()
    m["blind"] = m["blind"].str.strip().str.lower()

    print("=" * 72)
    print("S3 BLIND RELABEL vs ORIGINAL S3 LABELS")
    print("=" * 72)
    print("\nLabel counts:")
    print(pd.DataFrame({
        "original": m["original"].value_counts().reindex(LABELS, fill_value=0),
        "blind": m["blind"].value_counts().reindex(LABELS, fill_value=0),
    }).to_string())

    kappa, po = cohen_kappa(m["original"], m["blind"])
    print(f"\nExact agreement: {po:.3f} ({int(round(po * len(m)))}/{len(m)})")
    print(f"Cohen's kappa (4 labels): {kappa:.3f}")
    print(f"Strict correct: original {(m['original'] == 'correct').sum()}/29, "
          f"blind {(m['blind'] == 'correct').sum()}/29")

    print("\nCross-tab (rows = original, columns = blind):")
    print(pd.crosstab(m["original"], m["blind"])
          .reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())

    changed = m[m["original"] != m["blind"]]
    print(f"\nChanged labels: {len(changed)}")
    for _, r in changed.sort_values("question_id").iterrows():
        print(f"  {r['question_id']:<8} {r['original']:>10} -> {r['blind']:<10} {r['notes'] or '(no note)'}")


if __name__ == "__main__":
    try:
        finished = label_loop()
    except KeyboardInterrupt:
        finished = False
        print("\nStopped. Every label entered so far is saved.")
    if finished:
        clear()
        agreement()
    else:
        left = (df["correctness"].str.strip() == "").sum()
        print(f"\n{left} row(s) still to label. Rerun: python relabel_s3_blind.py")