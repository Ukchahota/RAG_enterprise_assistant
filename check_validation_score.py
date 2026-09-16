"""
Which system did each detector-validation claim come from, and was any
evidence shown for it?

Put this file in the repo root and run:
    python check_validation_source.py

Reads files only. Writes validation_source.txt - paste or upload it.
"""

import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
buf = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=buf)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_validation():
    names = ["detector_validation_human_final.csv", "detector_validation_scored.csv",
             "detector_validation_human_pass2.csv", "detector_validation_sample.csv"]
    for n in names + sorted(p.name for p in RES.glob("detector_validation*.csv")):
        p = RES / n
        if p.exists():
            d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if {"sentence", "predicted_label", "question_id"} <= set(d.columns):
                return p.name, d
    sys.exit("No validation file with question_id, sentence and predicted_label found.")


def main():
    vname, val = load_validation()
    cpath = RES / "hallucination_claims.csv"
    if not cpath.exists():
        sys.exit("results/hallucination_claims.csv not found")
    claims = pd.read_csv(cpath, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    say(f"Validation file: {vname} ({len(val)} rows)")
    say(f"Claims file: hallucination_claims.csv ({len(claims)} rows)")
    say("Claims file label counts by system:")
    say(pd.crosstab(claims["system"], claims["label"]).to_string())

    claims["key"] = claims["question_id"].str.strip() + "||" + claims["sentence"].map(norm)
    val["key"] = val["question_id"].str.strip() + "||" + val["sentence"].map(norm)
    lookup = claims.drop_duplicates("key").set_index("key")
    dup_keys = claims["key"].duplicated(keep=False)
    ambiguous = set(claims.loc[dup_keys, "key"])

    val["system"] = val["key"].map(lookup["system"]).fillna("UNMATCHED")
    val.loc[val["key"].isin(ambiguous), "system"] = val.loc[val["key"].isin(ambiguous), "key"].map(
        lambda k: "/".join(sorted(claims.loc[claims["key"] == k, "system"].unique())))

    chunk_col = "supporting_chunk_text" if "supporting_chunk_text" in val.columns else None
    if chunk_col:
        val["evidence_shown"] = val[chunk_col].str.strip().ne("") & val[chunk_col].str.lower().ne("nan")
    else:
        val["evidence_shown"] = val.get("supporting_chunk", pd.Series("", index=val.index)).str.strip().ne("")

    human = next((c for c in ["human_label", "human_label_pass2", "human"] if c in val.columns), None)

    say("\nSOURCE SYSTEM BY PREDICTED LABEL")
    say(pd.crosstab(val["predicted_label"], val["system"], margins=True).to_string())

    say("\nEVIDENCE TEXT SHOWN TO THE ANNOTATOR, BY PREDICTED LABEL")
    say(pd.crosstab(val["predicted_label"], val["evidence_shown"], margins=True).to_string())

    if human:
        say(f"\nPRECISION BY PREDICTED LABEL AND SOURCE SYSTEM (human column: {human})")
        v = val.copy()
        v["agree"] = v["predicted_label"].str.strip() == v[human].str.strip()
        entailed = {"supported", "supported_uncited"}
        v["agree_collapsed"] = v.apply(
            lambda r: (r["predicted_label"] in entailed and r[human] in entailed) or
                      (r["predicted_label"] == r[human]), axis=1)
        say(v.groupby(["predicted_label", "system"])["agree_collapsed"].agg(["sum", "count"]).to_string())

        say("\nTHE UNSUPPORTED STRATUM, ROW BY ROW")
        cols = [c for c in ["question_id", "system", "evidence_shown", "best_entailment",
                            "best_contradiction", human, "sentence"] if c in v.columns]
        u = v[v["predicted_label"] == "unsupported"][cols].copy()
        if "sentence" in u.columns:
            u["sentence"] = u["sentence"].str.slice(0, 70)
        say(u.to_string(index=False))

    (ROOT / "validation_source.txt").write_text(buf.getvalue(), encoding="utf-8")
    print("\nSaved validation_source.txt")


if __name__ == "__main__":
    main()