"""
Interactive labelling for S1 correctness on the stratified sample.

Shows one question at a time with the expected answer and S1's answer, and
writes each verdict straight to disk - so the session can be interrupted and
resumed without losing work. Already-labelled rows are skipped.

    python label_correctness.py                 # label unlabelled rows
    python label_correctness.py --refused-only  # the 13 that carry the result
    python label_correctness.py --review        # revisit labelled rows
    python label_correctness.py --summary       # tables, no labelling

Verdicts: c=correct  p=partial  i=incorrect  s=skip  q=quit
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

PATH = Path(r"results\correctness_s1_to_label.csv")

VERDICTS = {"c": "correct", "p": "partial", "i": "incorrect"}


def wrap(text, width=None):
    width = width or min(shutil.get_terminal_size((100, 25)).columns - 2, 100)
    import textwrap
    out = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return "\n".join(out)


def rule(char="=", label=""):
    width = min(shutil.get_terminal_size((100, 25)).columns - 2, 100)
    if label:
        return f"{char * 3} {label} {char * max(0, width - len(label) - 5)}"
    return char * width


def summary(df):
    print("\n" + rule("="))
    print("S1 CORRECTNESS BY WHAT S3 DID")
    print(rule("="))
    labelled = df[df["correctness"].notna() &
                  (df["correctness"].astype(str).str.strip() != "")]
    if labelled.empty:
        print("Nothing labelled yet.")
        return
    tab = pd.crosstab(labelled["s3_action"], labelled["correctness"])
    print(tab.to_string())

    ref = labelled[labelled["s3_action"] == "refused"]
    if not ref.empty:
        n_ok = int((ref["correctness"] == "correct").sum())
        n_part = int((ref["correctness"] == "partial").sum())
        print(f"\nCOVERAGE COST: of {len(ref)} questions S3 withheld, S1 "
              f"answered {n_ok} correctly")
        print(f"({n_ok / len(ref):.1%}){f', plus {n_part} partial' if n_part else ''}.")
        print("These are correct answers the corrective layer denied the user.")
        print(f"Sample is {len(ref)} of 45 total refusals, so report the "
              f"proportion with its denominator\nrather than extrapolating.")

    acc = labelled[labelled["s3_action"] == "accepted"]
    if not acc.empty:
        n_ok = int((acc["correctness"] == "correct").sum())
        print(f"\nMATCHED COMPARISON: on the {len(acc)} questions S3 answered, "
              f"S1 was correct on {n_ok}")
        print("Compare against the S3 correctness scores for the same items.")

    todo = len(df) - len(labelled)
    if todo:
        print(f"\n{todo} row(s) still unlabelled.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refused-only", action="store_true")
    p.add_argument("--review", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if not PATH.exists():
        raise SystemExit(f"Missing {PATH}")

    df = pd.read_csv(PATH, encoding="utf-8-sig", dtype=str)
    for col in ("correctness", "notes"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    if args.summary:
        summary(df)
        return

    mask = df["correctness"].astype(str).str.strip() != ""
    todo = df[mask].index if args.review else df[~mask].index
    if args.refused_only:
        todo = [i for i in todo if df.at[i, "s3_action"] == "refused"]

    if len(todo) == 0:
        print("Nothing to label.")
        summary(df)
        return

    print(f"{len(todo)} row(s) to label.  "
          f"c=correct  p=partial  i=incorrect  s=skip  q=quit\n")

    for n, i in enumerate(todo, start=1):
        row = df.loc[i]
        print("\n" + rule("="))
        print(f"[{n}/{len(todo)}]  {row['question_id']}  "
              f"({row['question_type']}, {row['difficulty']})   "
              f"S3: {row['s3_action'].upper()}")
        print(rule("="))
        print("\n" + rule("-", "QUESTION"))
        print(wrap(row["question"]))
        print("\n" + rule("-", "EXPECTED"))
        print(wrap(row["expected_answer"]))
        print("\n" + rule("-", "S1 ANSWER"))
        print(wrap(row["s1_answer"]))
        if str(row["correctness"]).strip():
            print(f"\n(current label: {row['correctness']})")

        while True:
            choice = input("\nverdict [c/p/i/s/q]: ").strip().lower()
            if choice in VERDICTS:
                df.at[i, "correctness"] = VERDICTS[choice]
                note = input("note (optional, Enter to skip): ").strip()
                if note:
                    df.at[i, "notes"] = note
                df.to_csv(PATH, index=False, encoding="utf-8-sig")
                print(f"  -> {VERDICTS[choice]}  (saved)")
                break
            if choice == "s":
                print("  -> skipped")
                break
            if choice == "q":
                print("\nStopping. Progress saved.")
                summary(df)
                return
            print("  Enter c, p, i, s or q.")

    print("\nDone.")
    summary(df)


if __name__ == "__main__":
    main()