"""
Paired correctness comparison: S0 (LLM-only) vs S1 (Basic RAG) vs S3 (Corrective RAG)
on the same 29 stratified questions.

Put this file in the repo root and run:
    python compare_correctness.py

Writes: results/correctness_all_systems.csv  (merged, derived - safe to regenerate)
"""

import sys
from math import comb, sqrt
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

FILES = {
    "s0": RES / "correctness_s0_to_label.csv",
    "s1": RES / "correctness_s1_to_label.csv",
    "s3": RES / "correctness_scores.csv",
}
LABELS = ["correct", "partial", "incorrect", "refused"]
SYSTEMS = ["s0", "s1", "s3"]
PAIRS = [("s0", "s1"), ("s0", "s3"), ("s1", "s3")]


def load(name, path):
    if not path.exists():
        sys.exit(f"Missing file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["question_id"] = df["question_id"].str.strip()
    df["correctness"] = df["correctness"].str.strip().str.lower()
    bad = df[~df["correctness"].isin(LABELS)]
    if len(bad):
        sys.exit(f"{name}: invalid labels\n{bad[['question_id', 'correctness']]}")
    if df["question_id"].duplicated().any():
        sys.exit(f"{name}: duplicate question_ids")
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    data = {s: load(s, p) for s, p in FILES.items()}

    ids = {s: set(d["question_id"]) for s, d in data.items()}
    if not (ids["s0"] == ids["s1"] == ids["s3"]):
        for s in SYSTEMS:
            print(f"{s}: {len(ids[s])} ids; not in s0: {sorted(ids[s] - ids['s0'])}")
        sys.exit("Question sets differ - cannot run a paired comparison.")

    s0 = data["s0"]
    s1 = data["s1"]
    merged = s0[["question_id", "question_type", "difficulty"]].copy()
    merged = merged.merge(s1[["question_id", "s3_action"]], on="question_id", how="left")
    for s in SYSTEMS:
        merged = merged.merge(
            data[s][["question_id", "correctness"]].rename(columns={"correctness": s}),
            on="question_id",
        )
    merged = merged.sort_values("question_id").reset_index(drop=True)
    n = len(merged)

    out = RES / "correctness_all_systems.csv"
    merged.to_csv(out, index=False, encoding="utf-8")

    # 1. Label distribution
    section(f"1. LABEL COUNTS  (n = {n} paired questions)")
    counts = pd.DataFrame(
        {s.upper(): merged[s].value_counts().reindex(LABELS, fill_value=0) for s in SYSTEMS}
    )
    print(counts.to_string())

    # 2. Accuracy under three partial-credit rules
    section("2. ACCURACY UNDER DIFFERENT PARTIAL-CREDIT RULES")
    rules = {"strict (partial=0)": 0.0, "half (partial=0.5)": 0.5, "lenient (partial=1)": 1.0}
    for rule, w in rules.items():
        print(f"\n{rule}")
        for s in SYSTEMS:
            score = (merged[s] == "correct").sum() + w * (merged[s] == "partial").sum()
            line = f"  {s.upper()}: {score:5.1f}/{n}  = {score / n:.3f}"
            if w in (0.0, 1.0):
                lo, hi = wilson(int(score), n)
                line += f"   95% Wilson CI [{lo:.3f}, {hi:.3f}]"
            print(line)

    # 3. Paired exact McNemar tests (binary rules only)
    section("3. PAIRED EXACT McNEMAR TESTS  (Holm-adjusted across the 3 pairs)")
    for rule, w in [("strict (partial=0)", 0.0), ("lenient (partial=1)", 1.0)]:
        ok = ["correct"] + (["partial"] if w == 1.0 else [])
        binary = {s: merged[s].isin(ok) for s in SYSTEMS}
        rows = []
        for a, b in PAIRS:
            only_a = int((binary[a] & ~binary[b]).sum())
            only_b = int((~binary[a] & binary[b]).sum())
            both = int((binary[a] & binary[b]).sum())
            neither = int((~binary[a] & ~binary[b]).sum())
            rows.append([f"{a.upper()} vs {b.upper()}", both, only_a, only_b, neither,
                         exact_mcnemar(only_a, only_b)])
        adj = holm([r[-1] for r in rows])
        print(f"\n{rule}")
        print(f"  {'pair':<12}{'both':>6}{'A only':>8}{'B only':>8}{'neither':>9}"
              f"{'p exact':>11}{'p Holm':>10}")
        for r, pa in zip(rows, adj):
            print(f"  {r[0]:<12}{r[1]:>6}{r[2]:>8}{r[3]:>8}{r[4]:>9}{r[5]:>11.4f}{pa:>10.4f}")

    # 4. Label-by-label cross-tabs
    section("4. CROSS-TABS  (rows = first system, columns = second)")
    for a, b in PAIRS:
        print(f"\n{a.upper()} x {b.upper()}")
        ct = pd.crosstab(merged[a], merged[b]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        print(ct.to_string())

    # 5. By S3 corrective action
    section("5. S0 AND S1 LABELS SPLIT BY S3 ACTION")
    if merged["s3_action"].str.strip().ne("").any():
        for s in ["s0", "s1"]:
            print(f"\n{s.upper()}")
            ct = pd.crosstab(merged["s3_action"], merged[s]).reindex(columns=LABELS, fill_value=0)
            print(ct.to_string())
    else:
        print("  s3_action column is empty - skipped")

    # 6. By question type (strict)
    section("6. STRICT CORRECT COUNTS BY QUESTION TYPE")
    by_type = merged.groupby("question_type").agg(
        n=("question_id", "size"),
        S0=("s0", lambda x: (x == "correct").sum()),
        S0_partial=("s0", lambda x: (x == "partial").sum()),
        S1=("s1", lambda x: (x == "correct").sum()),
        S3=("s3", lambda x: (x == "correct").sum()),
        S3_refused=("s3", lambda x: (x == "refused").sum()),
    )
    print(by_type.to_string())

    # 7. Audit list: every S0 partial with its note
    section("7. S0 PARTIAL LABELS  (audit these - they drive the RQ1 result)")
    partial = s0[s0["correctness"] == "partial"].sort_values("question_id")
    for _, r in partial.iterrows():
        note = r.get("notes", "").strip() or "(no note)"
        s1_label = merged.loc[merged["question_id"] == r["question_id"], "s1"].iloc[0]
        print(f"  {r['question_id']:<8} {r['question_type']:<28} S1={s1_label:<10} {note}")

    print(f"\nMerged table written to {out}")


if __name__ == "__main__":
    main()