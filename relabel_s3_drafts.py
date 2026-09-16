"""
Second labelling pass of S3's drafts for the 13 withheld questions.

Put this file in the repo root and run:
    python relabel_s3_drafts_pass2.py

First run : preserves pass 1 as results/correctness_s3_drafts_pass1.csv, builds
            results/correctness_s3_drafts_pass2.csv (blank labels, new order), then pauses.
Later runs: resumes labelling.
Labelling : every label needs a note.
              correct            -> the key fact as it appears in the draft
              partial/incorrect  -> which key fact is missing or wrong
When done : pass 1 vs pass 2 agreement, gate precision (pass 2), and why the gate fired.
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

PASS1_SRC = RES / "correctness_s3_drafts_blind.csv"
PASS1 = RES / "correctness_s3_drafts_pass1.csv"
SHEET = RES / "correctness_s3_drafts_pass2.csv"
S3_BLIND = RES / "correctness_s3_relabel_blind.csv"
SAMPLE = RES / "correctness_sample.csv"
ANSWER_FILES = [RES / "answers_s3_corrective_rag_clean.csv", RES / "answers_s3_corrective_rag.csv"]

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
    if not PASS1.exists():
        shutil.copyfile(read_path(PASS1_SRC), PASS1)
        print(f"Pass 1 preserved as {PASS1.name}")
    p1 = read(PASS1)
    if (p1["correctness"].str.strip() == "").any():
        sys.exit("Pass 1 has blank labels - finish it before starting pass 2.")
    sheet = p1[["question_id", "question_type", "question", "expected_answer", "s3_draft"]].copy()
    sheet = sheet.sample(frac=1, random_state=41).reset_index(drop=True)
    sheet["correctness"] = ""
    sheet["notes"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8")
    print(f"Built {SHEET.name} ({len(sheet)} drafts, blank labels, new order with seed 41).")
    print("\nCommit pass 1 and the empty pass 2 sheet before labelling:")
    print(f'  git add "{PASS1.relative_to(ROOT)}" "{PASS1_SRC.relative_to(ROOT)}" '
          f'"{SHEET.relative_to(ROOT)}" relabel_s3_drafts_pass2.py')
    print('  git commit -m "S3 drafts: pass 1 preserved, pass 2 sheet before labelling"')
    print("\nDo not open the pass 1 file while labelling.")
    input("\nPress Enter to start labelling (or Ctrl+C to stop and commit first)...")


def read_path(p):
    if not p.exists():
        sys.exit(f"Missing {p}")
    return p


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


def ask_note(prompt):
    note = ""
    while not note:
        note = input(prompt).strip()
    return note


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
        print("Check EVERY key fact in the expected answer against the draft before choosing.")
        print("Ignore citation markers like [1].\n")

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
            if choice == "c":
                note = ask_note("Key fact(s) as they appear in the draft (required): ")
            elif choice in ("p", "i"):
                note = ask_note("Which key fact is missing or wrong? (required): ")
            else:
                note = input("Note (optional): ").strip()
            history.append((k, df.at[k, "notes"]))
            df.at[k, "correctness"] = KEYS[choice]
            df.at[k, "notes"] = note
            save()
        else:
            input("Unrecognised key - press Enter to try again")


def kappa(a, b):
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in LABELS)
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po


def load_diagnostics(ids):
    sample = read(SAMPLE)
    sample_ids = set(sample["question_id"])
    for f in ANSWER_FILES:
        if not f.exists():
            continue
        ans = read(f).drop_duplicates("question_id").set_index("question_id")
        ok = all(q in ans.index and norm(ans.at[q, "answer"]) ==
                 norm(sample.loc[sample["question_id"] == q, "answer"].iloc[0]) for q in sample_ids)
        if ok:
            return ans.loc[ids], f.name
    return None, None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def summary():
    p2 = df.copy()
    p2["question_id"] = p2["question_id"].str.strip()
    p2["pass2"] = p2["correctness"].str.strip().str.lower()
    p1 = read(PASS1)
    p1_lab = dict(zip(p1["question_id"], p1["correctness"].str.strip().str.lower()))
    p2["pass1"] = p2["question_id"].map(p1_lab)
    n = len(p2)

    print("=" * 72)
    print(f"S3 DRAFTS, PASS 2 (n = {n})")
    print("=" * 72)
    print(pd.DataFrame({
        "pass1": p2["pass1"].value_counts().reindex(LABELS, fill_value=0),
        "pass2": p2["pass2"].value_counts().reindex(LABELS, fill_value=0),
    }).to_string())
    k, po = kappa(p2["pass1"], p2["pass2"])
    print(f"\nPass 1 vs pass 2 exact agreement: {po:.3f} ({int(round(po * n))}/{n}), kappa {k:.3f}"
          "  (kappa is undefined or unstable when one pass uses a single label)")

    lost = int((p2["pass2"] == "correct").sum())
    print("\nGATE AGAINST S3'S OWN DRAFTS (pass 2)")
    print(f"  correct drafts withheld (cost):        {lost}/{n}")
    print(f"  non-correct drafts withheld (benefit): {n - lost}/{n}  = gate precision {(n - lost) / n:.3f}")

    s3 = read(S3_BLIND)
    delivered = int((s3["correctness"].str.strip().str.lower() == "correct").sum())
    print(f"\n  S3 gated: {delivered}/29   ungated: {delivered + lost}/29"
          "  (ungated assumes accepted answers equal their drafts)")

    diag, src = load_diagnostics(p2["question_id"].tolist())
    print("\n" + "=" * 72)
    print("WHY THE GATE FIRED" + (f"  (from {src})" if src else "  (no matching answers file found)"))
    print("=" * 72)
    if diag is not None:
        print("Trigger rule: any contradiction OR draft unsupported rate > 0.40\n")
        trig_counts = {"contradiction only": 0, "unsupported only": 0, "both": 0, "neither (check)": 0}
        print(f"  {'id':<8}{'pass2':<11}{'unsup':>7}{'sup':>5}{'uns':>5}{'con':>5}"
              f"{'gold_ev':>9}  trigger / correction_reason")
        for _, r in p2.sort_values("question_id").iterrows():
            d = diag.loc[r["question_id"]]
            rate = to_float(d.get("draft_unsupported_rate"))
            ncon = to_float(d.get("draft_n_contradicted"))
            gold = to_float(d.get("gold_in_evidence"))
            con, uns = ncon > 0, rate > 0.40
            key = ("both" if con and uns else "contradiction only" if con
                   else "unsupported only" if uns else "neither (check)")
            trig_counts[key] += 1
            print(f"  {r['question_id']:<8}{r['pass2']:<11}{rate:>7.2f}"
                  f"{d.get('draft_n_supported', ''):>5}{d.get('draft_n_unsupported', ''):>5}"
                  f"{d.get('draft_n_contradicted', ''):>5}{gold:>9.2f}  "
                  f"{key} / {d.get('correction_reason', '')}")
        print("\nTrigger counts:")
        for t, c in trig_counts.items():
            print(f"  {t:<20} {c}")
        complete = sum(to_float(diag.loc[q].get("gold_in_evidence")) >= 1.0 for q in p2["question_id"])
        print(f"\nComplete retrieval (gold_in_evidence >= 1.0): {complete}/{n}")

    print("\nPer question (pass 2 notes):")
    for _, r in p2.sort_values("question_id").iterrows():
        flag = "" if r["pass1"] == r["pass2"] else f"  [pass1={r['pass1']}]"
        print(f"  {r['question_id']:<8} {r['pass2']:<10} {r['notes']}{flag}")


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
        print(f"\n{left} row(s) still to label. Rerun: python relabel_s3_drafts_pass2.py")