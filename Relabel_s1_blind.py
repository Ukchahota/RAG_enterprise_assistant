"""
Blind re-labelling of S1 correctness under the committed rubric.

Put this file in the repo root and run:
    python relabel_s1_blind.py

First run : builds results/correctness_s1_relabel_blind.csv, then pauses so you can commit it.
Later runs: resumes labelling from the first blank row.
When done : prints agreement with the original S1 labels (never modifies the original file).
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

ORIGINAL = RES / "correctness_s1_to_label.csv"
SHEET = RES / "correctness_s1_relabel_blind.csv"
KEYS = {"c": "correct", "p": "partial", "i": "incorrect", "r": "refused"}
LABELS = ["correct", "partial", "incorrect", "refused"]
RUBRIC = ("Rubric: correct = all key facts, no contradicting specifics | "
          "partial = some key facts (scores zero) | "
          "incorrect = wrong specifics or generic answer lacking the Liverpool-specific fact | "
          "refused = declines / says it cannot know")


def build_sheet():
    if not ORIGINAL.exists():
        sys.exit(f"Missing {ORIGINAL}")
    src = pd.read_csv(ORIGINAL, dtype=str, keep_default_na=False)
    keep = ["question_id", "question_type", "difficulty", "question", "expected_answer", "s1_answer"]
    sheet = src[keep].copy()
    assert len(sheet) == 29 and sheet["question_id"].is_unique, "expected 29 unique questions"
    sheet = sheet.sample(frac=1, random_state=7).reset_index(drop=True)
    sheet["correctness"] = ""
    sheet["notes"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8")
    print(f"Built {SHEET} (29 rows, s3_action and original labels removed, shuffled with seed 7).")
    print("\nCommit it before labelling so the audit trail shows it was created first:")
    print(f'  git add "{SHEET.relative_to(ROOT)}" relabel_s1_blind.py')
    print('  git commit -m "S1 blind relabel sheet, before labelling"')
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
        print(f"[{len(df) - len(todo)}/{len(df)} labelled]  {row['question_id']}  "
              f"({row['question_type']}, {row['difficulty']})\n")
        show("QUESTION:", row["question"])
        show("EXPECTED ANSWER:", row["expected_answer"])
        show("S1 ANSWER:", row["s1_answer"])
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
    m = orig[["question_id", "correctness"]].rename(columns={"correctness": "original"}).merge(
        df[["question_id", "correctness", "notes"]].rename(columns={"correctness": "blind"}),
        on="question_id")
    m["original"] = m["original"].str.strip().str.lower()
    m["blind"] = m["blind"].str.strip().str.lower()

    print("=" * 72)
    print("S1 BLIND RELABEL vs ORIGINAL S1 LABELS")
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
        print(f"\n{left} row(s) still to label. Rerun: python relabel_s1_blind.py")