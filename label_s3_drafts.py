"""
Blind labelling of S3's own draft answers for the questions S3 withheld.

This measures the corrective gate against S3's own counterfactual
(what it would have said without withholding), instead of using S1 as a proxy.

Put this file in the repo root and run:
    python label_s3_drafts.py

First run : finds the S3 answers file that matches the labelled S3 answers, builds
            results/correctness_s3_drafts_blind.csv, then pauses so you can commit it.
Later runs: resumes labelling.
When done : prints draft labels, gate precision, ungated S3 accuracy, and S1-proxy agreement.
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

S3_BLIND = RES / "correctness_s3_relabel_blind.csv"
S1_BLIND = RES / "correctness_s1_relabel_blind.csv"
SAMPLE = RES / "correctness_sample.csv"
ANSWER_FILES = [RES / "answers_s3_corrective_rag_clean.csv", RES / "answers_s3_corrective_rag.csv"]
SHEET = RES / "correctness_s3_drafts_blind.csv"

KEYS = {"c": "correct", "p": "partial", "i": "incorrect", "r": "refused"}
LABELS = ["correct", "partial", "incorrect", "refused"]
RUBRIC = ("Rubric: correct = all key facts, no contradicting specifics | "
          "partial = some key facts (scores zero) | "
          "incorrect = wrong specifics or generic answer lacking the Liverpool-specific fact | "
          "refused = declines / says it cannot know")


def norm(s):
    return " ".join(str(s).split())


def read(p):
    if not p.exists():
        sys.exit(f"Missing {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["question_id"] = df["question_id"].str.strip()
    return df


def build_sheet():
    s3 = read(S3_BLIND)
    sample = read(SAMPLE)
    withheld = sorted(s3.loc[s3["correctness"].str.strip().str.lower() == "refused", "question_id"])
    ids = set(sample["question_id"])
    print(f"Withheld questions in the blind S3 labels: {len(withheld)}")

    chosen = None
    for f in ANSWER_FILES:
        if not f.exists():
            continue
        ans = read(f).drop_duplicates("question_id").set_index("question_id")
        if "draft_answer" not in ans.columns:
            print(f"{f.name}: no draft_answer column")
            continue
        ok = all(q in ans.index and norm(ans.at[q, "answer"]) ==
                 norm(sample.loc[sample["question_id"] == q, "answer"].iloc[0]) for q in ids)
        print(f"{f.name}: answers {'match' if ok else 'do NOT match'} the labelled S3 answers")
        if ok:
            chosen, table = f, ans
            break
    if chosen is None:
        sys.exit("No S3 answers file matches the labelled answers. Paste this output to Claude.")

    rows = table.loc[withheld]
    empty = [q for q in withheld if not norm(rows.at[q, "draft_answer"])]
    if empty:
        sys.exit(f"Empty draft_answer for: {empty}. Paste this output to Claude.")
    same = [q for q in withheld if norm(rows.at[q, "draft_answer"]) == norm(rows.at[q, "answer"])]
    if same:
        print(f"Warning: draft identical to delivered answer for {same}")

    sheet = sample.set_index("question_id").loc[withheld, ["question_type", "question", "expected_answer"]]
    sheet["s3_draft"] = rows["draft_answer"]
    sheet = sheet.reset_index().sample(frac=1, random_state=23).reset_index(drop=True)
    sheet["correctness"] = ""
    sheet["notes"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8")
    print(f"\nBuilt {SHEET.name} from {chosen.name} ({len(sheet)} drafts, shuffled with seed 23).")
    print("\nCommit it before labelling:")
    print(f'  git add "{SHEET.relative_to(ROOT)}" label_s3_drafts.py')
    print('  git commit -m "S3 draft labelling sheet, before labelling"')
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
            input("File is locked (open in Excel, or OneDrive syncing). Close it, then press Enter...")


def show(label, text):
    width = max(60, shutil.get_terminal_size().columns - 4)
    print(label)
    for para in str(text).splitlines():
        print(textwrap.fill(para, width, initial_indent="  ", subsequent_indent="  ")
              if para.strip() else "")
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
        show("DRAFT ANSWER:", row["s3_draft"])
        print(textwrap.fill(RUBRIC, max(60, shutil.get_terminal_size().columns - 4)) + "\n")
        print("Judge the draft's content. Citation markers like [1] can be ignored.")
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
            if choice in ("p", "i"):
                note = ""
                while not note:
                    note = input("Which key fact is missing or wrong? (required): ").strip()
            else:
                note = input("Note (optional): ").strip()
            history.append((k, df.at[k, "notes"]))
            df.at[k, "correctness"] = KEYS[choice]
            df.at[k, "notes"] = note
            save()
        else:
            input("Unrecognised key - press Enter to try again")


def summary():
    d = df.copy()
    d["question_id"] = d["question_id"].str.strip()
    d["draft"] = d["correctness"].str.strip().str.lower()
    n = len(d)
    s3 = read(S3_BLIND)
    s1 = read(S1_BLIND)
    s3_lab = s3["correctness"].str.strip().str.lower()
    s1_lab = dict(zip(s1["question_id"], s1["correctness"].str.strip().str.lower()))
    delivered_correct = int((s3_lab == "correct").sum())

    print("=" * 72)
    print(f"S3 DRAFTS FOR THE {n} WITHHELD QUESTIONS (blind)")
    print("=" * 72)
    print(d["draft"].value_counts().reindex(LABELS, fill_value=0).to_string())

    lost = int((d["draft"] == "correct").sum())
    avoided = n - lost
    print("\nGATE AGAINST S3'S OWN DRAFTS")
    print(f"  correct drafts withheld (cost):          {lost}/{n}")
    print(f"  non-correct drafts withheld (benefit):   {avoided}/{n}"
          f"  = gate precision {avoided / n:.3f}")
    print(f"    of which incorrect: {int((d['draft'] == 'incorrect').sum())}, "
          f"partial: {int((d['draft'] == 'partial').sum())}, "
          f"draft already a refusal: {int((d['draft'] == 'refused').sum())}")

    print("\nS3 WITH vs WITHOUT THE GATE (strict, n = 29)")
    print(f"  gated (as deployed):  {delivered_correct}/29 = {delivered_correct / 29:.3f}")
    print(f"  ungated (drafts used): {delivered_correct + lost}/29 = {(delivered_correct + lost) / 29:.3f}")
    print("  (ungated assumes accepted answers are unchanged from their drafts)")

    d["s1"] = d["question_id"].map(s1_lab)
    agree = (d["s1"] == d["draft"]).mean()
    print(f"\nS1 AS A PROXY FOR THE DRAFTS: exact agreement {agree:.3f} "
          f"({int(round(agree * n))}/{n})")
    print(pd.crosstab(d["s1"], d["draft"], rownames=["S1 blind"], colnames=["S3 draft"])
          .reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())

    print("\nPer question:")
    for _, r in d.sort_values("question_id").iterrows():
        print(f"  {r['question_id']:<8} {r['question_type']:<28} draft={r['draft']:<10} "
              f"S1={r['s1']:<10} {r['notes']}")


if __name__ == "__main__":
    try:
        finished = label_loop()
    except KeyboardInterrupt:
        finished = False
        print("\nStopped. Every label entered so far is saved.")
    if finished:
        clear()
        summary()
    else:
        left = (df["correctness"].str.strip() == "").sum()
        print(f"\n{left} row(s) still to label. Rerun: python label_s3_drafts.py")