"""
Interactive S0 correctness labeller.

Put this file in the repo root (next to the results folder) and run:
    python label_s0.py

Shows one question at a time and saves after every label.
Quit whenever you like and rerun to continue from the first unlabelled row.
"""

import os
import sys
import shutil
import textwrap
from pathlib import Path

import pandas as pd

# Print answer text safely on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RELATIVE = Path("results") / "correctness_s0_to_label.csv"
KEYS = {"c": "correct", "p": "partial", "i": "incorrect", "r": "refused"}
ALLOWED = set(KEYS.values())
RUBRIC = ("Rubric: generic answer lacking the Liverpool-specific key fact = incorrect. "
          "Partial scores zero.")


def find_sheet():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        candidate = base / RELATIVE
        if candidate.exists():
            return candidate
    sys.exit(f"Could not find {RELATIVE}. Put this script in the repo root "
             f"(the folder that contains 'results') and run it from there.")


PATH = find_sheet()
df = pd.read_csv(PATH, dtype=str, keep_default_na=False)
for col in ("correctness", "notes"):
    if col not in df.columns:
        df[col] = ""


def save():
    tmp = PATH.with_name(PATH.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    while True:
        try:
            os.replace(tmp, PATH)
            return
        except PermissionError:
            input("File is locked (open in Excel, or OneDrive syncing). "
                  "Close it, then press Enter to retry...")


def show(label, text):
    width = max(60, shutil.get_terminal_size().columns - 4)
    lines = []
    for para in str(text).splitlines():
        if para.strip():
            lines.append(textwrap.fill(para, width,
                                       initial_indent="  ", subsequent_indent="  "))
        else:
            lines.append("")
    print(label)
    print("\n".join(lines))
    print()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def summary():
    labels = df["correctness"].str.strip().str.lower()
    print(f"\nrows: {len(df)}, unique ids: {df['question_id'].nunique()}")
    print(labels.replace("", "(blank)").value_counts().to_string())
    bad = df[~labels.isin(ALLOWED)]
    if len(bad):
        print(f"\n{len(bad)} row(s) still missing or invalid.")
    else:
        print("\nAll labels valid.")


def main():
    history = []
    skipped = set()
    force = None

    while True:
        todo = [k for k in df.index if not df.at[k, "correctness"].strip()]
        if not todo:
            clear()
            print("All rows labelled.")
            break

        if force is not None:
            k, force = force, None
        else:
            pending = [k for k in todo if k not in skipped]
            if not pending:
                skipped.clear()
                pending = todo
            k = pending[0]

        row = df.loc[k]
        done = len(df) - len(todo)
        clear()
        print(f"[{done}/{len(df)} labelled]  {row['question_id']}  "
              f"({row.get('question_type', '')}, {row.get('difficulty', '')})\n")
        show("QUESTION:", row["question"])
        show("EXPECTED ANSWER:", row["expected_answer"])
        show("S0 ANSWER:", row["s0_answer"])
        print(RUBRIC + "\n")

        choice = input("[c]orrect [p]artial [i]ncorrect [r]efused | "
                       "[s]kip [u]ndo [q]uit > ").strip().lower()

        if choice == "q":
            break
        elif choice == "s":
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
            note = input("Note (Enter to leave blank): ").strip()
            history.append((k, df.at[k, "notes"]))
            df.at[k, "correctness"] = KEYS[choice]
            df.at[k, "notes"] = note
            save()
        else:
            input("Unrecognised key - press Enter to try again")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Every label entered so far is saved.")
    summary()