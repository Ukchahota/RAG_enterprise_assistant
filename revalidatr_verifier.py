"""
Blind re-validation of the CURRENT verifier's claim labels on S1-S3 answers.

Put this file in the repo root and run:
    python revalidate_verifier.py

Optional  : python revalidate_verifier.py results\\<claims file>.csv  (choose the claims file)

First run : samples claims stratified by predicted label, writes
            results/verifier_revalidation_sheet.csv (what you see) and
            results/verifier_revalidation_key.csv   (hidden labels/scores),
            then pauses so you can commit both before labelling.
Later runs: resumes labelling one claim at a time.
When done : scores precision per predicted label and writes
            results/verifier_revalidation_scores.txt

You judge each claim ONLY against the passages shown (the evidence the system
actually retrieved for that answer):
  e = entailed      the passages state or directly imply the claim
  u = unsupported   the passages neither support nor contradict it
  c = contradicted  a passage states something incompatible with it
  n = not a claim   no checkable factual content (advice, hedge, meta-statement)
"""

import os
import re
import shutil
import sys
import textwrap
from math import sqrt
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- sample design (fixed before labelling) ----
PER_LABEL = {"unsupported": 20, "contradicted": 10, "entailed": 10, "nei": 5}
SEED = 2026
SYSTEMS = {
    "S1_basic_rag": "answers_s1_basic_rag.csv",
    "S2_modular_rag": "answers_s2_modular_rag.csv",
    "S3_corrective_rag": "answers_s3_corrective_rag_clean.csv",
}

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SHEET = RES / "verifier_revalidation_sheet.csv"
KEY = RES / "verifier_revalidation_key.csv"
SCORES = RES / "verifier_revalidation_scores.txt"
KEYS = {"e": "entailed", "u": "unsupported", "c": "contradicted", "n": "not_a_claim"}
HUMAN_LABELS = ["entailed", "unsupported", "contradicted", "not_a_claim"]


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


NEEDED = ["system", "question_id", "label", "sentence"]
OPTIONAL = ["question_type", "cited", "best_entailment", "best_contradiction", "supporting_chunk"]


def try_read_claims(p):
    """Read only the columns needed; return None (with a message) if the file is unusable."""
    try:
        with open(p, encoding="utf-8-sig", errors="replace") as fh:
            header = fh.readline()
        cols = [c.strip().strip('"') for c in header.strip().split(",")]
        if not set(NEEDED) <= set(cols):
            print(f"  skip {p.name}: missing columns {sorted(set(NEEDED) - set(cols))}")
            return None
        use = [c for c in NEEDED + OPTIONAL if c in cols]
        d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=use)
        if not d["system"].isin(SYSTEMS).any():
            print(f"  skip {p.name}: no S1-S3 rows (systems: {sorted(d['system'].unique())[:6]})")
            return None
        return d
    except Exception as e:
        print(f"  skip {p.name}: could not be read ({type(e).__name__}: {str(e)[:80]})")
        return None


def find_claims_file():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        p = p if p.is_absolute() else ROOT / p
        d = try_read_claims(p)
        if d is None:
            sys.exit(f"{p} is not a usable claims file.")
        return p, d
    files = sorted(RES.glob("hallucination_claims*.csv"))
    print("Claims files found:")
    for f in files:
        print(f"  {f.name:<55} {f.stat().st_size / 1_048_576:8.1f} MB")
    # Only files from the current verifier run qualify; older files (S0/S2 only,
    # or pre-rebuild S1/S3) are deliberately never used as a fallback.
    preferred = [f for f in files if "final_v2" in f.name] + \
                [f for f in files if "final" in f.name and "final_v2" not in f.name]
    print("\nTrying current-verifier files in order:")
    for p in preferred:
        d = try_read_claims(p)
        if d is None:
            continue
        present = set(d["system"]) & set(SYSTEMS)
        if present != set(SYSTEMS):
            print(f"  skip {p.name}: has only {sorted(present)}")
            continue
        return p, d
    sys.exit("No usable claims file with S1-S3 rows. Run: python revalidate_verifier.py results\\<file>.csv")


def evidence_lookup():
    meta_path = ROOT / "vector_store" / "chunk_metadata.csv"
    meta = read(meta_path)
    text = dict(zip(meta["chunk_id"], meta["chunk_text"]))
    doc = dict(zip(meta["chunk_id"], meta.get("document_name", pd.Series("", index=meta.index))))
    page = dict(zip(meta["chunk_id"], meta.get("page_number", pd.Series("", index=meta.index))))
    ids = set(text)
    out = {}
    for system, fname in SYSTEMS.items():
        p = RES / fname
        if not p.exists():
            print(f"warning: {fname} not found")
            continue
        a = read(p)
        col = next((c for c in ["evidence_chunk_ids", "retrieved_chunk_ids"] if c in a.columns), None)
        if col is None:
            print(f"warning: no evidence id column in {fname}")
            continue
        for _, r in a.iterrows():
            tokens = [t for t in re.split(r"[\s,;|\[\]'\"]+", str(r[col])) if t]
            chosen = [t for t in tokens if t in ids]
            if not chosen:
                continue
            parts = [f"[{i}] ({doc.get(c, '')}, page {page.get(c, '')})\n{text[c]}"
                     for i, c in enumerate(chosen, 1)]
            out[(system, r["question_id"].strip())] = "\n\n".join(parts)
    return out


def build():
    cpath, claims = find_claims_file()
    claims = claims[claims["system"].isin(SYSTEMS)].copy()
    claims["question_id"] = claims["question_id"].str.strip()
    claims["stratum"] = claims["label"].replace({"supported": "entailed", "supported_uncited": "entailed"})
    ev = evidence_lookup()
    claims["evidence"] = [ev.get((s, q), "") for s, q in zip(claims["system"], claims["question_id"])]
    claims = claims[claims["evidence"].str.strip() != ""]
    print(f"Claims file: {cpath.name}; S1-S3 claims with evidence: {len(claims)}")
    print(pd.crosstab(claims["stratum"], claims["system"]).to_string())

    frames = []
    for stratum, n in PER_LABEL.items():
        g = claims[claims["stratum"] == stratum]
        if g.empty:
            print(f"warning: no '{stratum}' claims available")
            continue
        take = min(n, len(g))
        # spread the draw across systems as evenly as the pool allows
        quota = -(-take // g["system"].nunique())
        per_sys = pd.concat([x.sample(n=min(len(x), quota), random_state=SEED)
                             for _, x in g.groupby("system")])
        if len(per_sys) < take:
            rest = g.drop(per_sys.index)
            per_sys = pd.concat([per_sys, rest.sample(n=min(len(rest), take - len(per_sys)), random_state=SEED)])
        frames.append(per_sys.sample(n=take, random_state=SEED))
    sample = pd.concat(frames).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["item_id"] = [f"RV{i:03d}" for i in range(1, len(sample) + 1)]

    qtext = {}
    for fname in SYSTEMS.values():
        p = RES / fname
        if p.exists():
            a = read(p)
            qtext.update(dict(zip(a["question_id"].str.strip(), a["question"])))

    sheet = pd.DataFrame({
        "item_id": sample["item_id"],
        "question": sample["question_id"].map(qtext).fillna(""),
        "claim": sample["sentence"],
        "evidence": sample["evidence"],
        "human_label": "",
        "notes": "",
    })
    keep = [c for c in ["item_id", "system", "question_id", "question_type", "label", "stratum",
                        "cited", "best_entailment", "best_contradiction", "supporting_chunk"] if c in sample.columns]
    sample[keep].to_csv(KEY, index=False, encoding="utf-8")
    sheet.to_csv(SHEET, index=False, encoding="utf-8")
    print(f"\nBuilt {SHEET.name} ({len(sheet)} claims) and {KEY.name} (hidden).")
    print("Sample by stratum:", sample["stratum"].value_counts().to_dict())
    print("Sample by system: ", sample["system"].value_counts().to_dict())
    print("\nCommit both files before labelling:")
    print(f'  git add "{SHEET.relative_to(ROOT)}" "{KEY.relative_to(ROOT)}" revalidate_verifier.py')
    print('  git commit -m "Verifier re-validation sample (S1-S3, current verifier), before labelling"')
    print("\nDo not open the key file while labelling.")
    input("\nPress Enter to start labelling (or Ctrl+C to stop and commit first)...")


def save(df):
    tmp = SHEET.with_name(SHEET.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    while True:
        try:
            os.replace(tmp, SHEET)
            return
        except PermissionError:
            input("File is locked (Excel or OneDrive). Close it, then press Enter...")


def show(label, text):
    width = max(60, shutil.get_terminal_size().columns - 4)
    print(label)
    for para in str(text).splitlines():
        print(textwrap.fill(para, width, initial_indent="  ", subsequent_indent="  ") if para.strip() else "")
    print()


def label_loop(df):
    history = []
    force = None
    while True:
        todo = [k for k in df.index if not df.at[k, "human_label"].strip()]
        if not todo:
            return True
        k = force if force is not None else todo[0]
        force = None
        r = df.loc[k]
        os.system("cls" if os.name == "nt" else "clear")
        print(f"[{len(df) - len(todo)}/{len(df)} labelled]  {r['item_id']}\n")
        show("QUESTION:", r["question"])
        show("EVIDENCE SHOWN TO THE SYSTEM:", r["evidence"])
        show("CLAIM TO JUDGE:", r["claim"])
        print("Judge the claim ONLY against the evidence above, not against your own knowledge.")
        print("  e = entailed | u = unsupported | c = contradicted | n = not a checkable claim\n")
        choice = input("[e/u/c/n] label | z = undo | q = quit > ").strip().lower()
        if choice == "q":
            return False
        if choice == "z":
            if history:
                last, old = history.pop()
                df.at[last, "human_label"] = ""
                df.at[last, "notes"] = old
                save(df)
                force = last
            else:
                input("Nothing to undo - press Enter")
            continue
        if choice not in KEYS:
            input("Unrecognised key - press Enter")
            continue
        note = ""
        if choice in ("c", "n"):
            while not note:
                note = input("Why? (required: the incompatible statement, or why it is not a claim): ").strip()
        else:
            note = input("Note (optional): ").strip()
        history.append((k, df.at[k, "notes"]))
        df.at[k, "human_label"] = KEYS[choice]
        df.at[k, "notes"] = note
        save(df)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(df):
    key = read(KEY)
    m = df.merge(key, on="item_id")
    expected = {"entailed": "entailed", "unsupported": "unsupported",
                "contradicted": "contradicted", "nei": "not_a_claim"}
    m["agree"] = m["human_label"] == m["stratum"].map(expected)
    lines = ["VERIFIER RE-VALIDATION (current verifier, S1-S3 claims, evidence shown)", ""]
    lines.append(f"Items: {len(m)}")
    lines.append("")
    lines.append("Precision by predicted label (95% Wilson interval):")
    for stratum in PER_LABEL:
        g = m[m["stratum"] == stratum]
        if g.empty:
            continue
        k, n = int(g["agree"].sum()), len(g)
        lo, hi = wilson(k, n)
        lines.append(f"  {stratum:<13} {k:>2}/{n:<2} = {k / n:.2f}   [{lo:.2f}, {hi:.2f}]")
    lines.append("")
    lines.append("Confusion (rows = verifier, columns = human):")
    lines.append(pd.crosstab(m["stratum"], m["human_label"]).reindex(columns=HUMAN_LABELS, fill_value=0).to_string())
    lines.append("")
    lines.append("Precision by predicted label and system:")
    lines.append(m.groupby(["stratum", "system"])["agree"].agg(["sum", "count"]).to_string())
    if "best_entailment" in m.columns:
        lines.append("")
        lines.append("Disagreements:")
        cols = [c for c in ["item_id", "system", "question_type", "label", "human_label",
                            "best_entailment", "best_contradiction", "notes"] if c in m.columns]
        lines.append(m[~m["agree"]][cols].to_string(index=False))
    text = "\n".join(lines)
    SCORES.write_text(text, encoding="utf-8")
    os.system("cls" if os.name == "nt" else "clear")
    print(text)
    print(f"\nSaved {SCORES.relative_to(ROOT)}")


if __name__ == "__main__":
    if not SHEET.exists():
        build()
    sheet = read(SHEET)
    try:
        done = label_loop(sheet)
    except KeyboardInterrupt:
        done = False
        print("\nStopped. Every label entered so far is saved.")
    if done:
        score(sheet)
    else:
        left = int((sheet["human_label"].str.strip() == "").sum())
        print(f"\n{left} claim(s) still to label. Rerun: python revalidate_verifier.py")