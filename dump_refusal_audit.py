"""
Builds a human-labelling pack for contradiction-triggered refusals.

For each selected question it writes the draft answer, the evidence the
verifier saw (marked gold / non-gold, with retrieval scores), the trigger
reason, and a blank verdict block. The judgement being made: was the draft
ACTUALLY contradicted by the evidence, or is this a false refusal?

    python dump_refusal_audit.py                  # 8 factual contradiction cases
    python dump_refusal_audit.py --all-contradiction   # all 25
    python dump_refusal_audit.py --all-refused         # all 45
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
ANSWERS = RESULTS / "answers_s3_corrective_rag.csv"

# Contradiction-triggered refusals on factual questions: complete retrieval
# on easy questions, so false refusals should concentrate here.
PRIORITY = ["V2Q013", "V2Q014", "V2Q021", "V2Q042",
            "V2Q048", "V2Q049", "V2Q061", "V2Q074"]


def split_ids(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def evidence_list(row):
    ids = split_ids(row.get("evidence_chunk_ids"))
    raw = row.get("evidence_texts", "")
    texts = []
    if isinstance(raw, str) and raw.startswith("["):
        try:
            texts = json.loads(raw)
        except json.JSONDecodeError:
            texts = []
    if len(texts) != len(ids):
        texts = [""] * len(ids)
    scores = split_ids(row.get("evidence_scores"))
    scores += [""] * (len(ids) - len(scores))
    return list(zip(ids, texts, scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all-contradiction", action="store_true")
    p.add_argument("--all-refused", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(ANSWERS, encoding="utf-8-sig")
    refused = df[df["corrective_action"] == "refused"]

    if args.all_refused:
        sel, tag = refused, "all_refused"
    elif args.all_contradiction:
        sel = refused[refused["correction_reason"].str.contains("contradicted",
                                                               na=False)]
        tag = "all_contradiction"
    else:
        sel, tag = refused[refused["question_id"].isin(PRIORITY)], "priority"

    out_md = RESULTS / f"refusal_audit_{tag}.md"
    rows = []
    lines = [f"# Refusal audit ({tag}) — {len(sel)} questions\n",
             "For each: was the draft genuinely contradicted by the evidence?\n",
             "Verdict options: TRUE_CONTRADICTION / FALSE_REFUSAL / UNCLEAR\n"]

    for _, r in sel.iterrows():
        ev = evidence_list(r)
        gold = set(split_ids(r.get("gold_chunk_ids")))

        lines.append(f"\n---\n\n## {r['question_id']} "
                     f"({r.get('question_type')}, {r.get('difficulty')})\n")
        lines.append(f"**Question:** {r['question']}\n")
        lines.append(f"**Expected answer:** {r.get('expected_answer')}\n")
        lines.append(f"**Trigger:** {r.get('correction_reason')}\n")
        lines.append(f"**gold_in_evidence:** {r.get('gold_in_evidence')}\n")
        lines.append(f"\n### Draft answer (withheld)\n\n{r.get('draft_answer')}\n")
        lines.append("\n### Evidence shown to the generator\n")
        for i, (cid, text, score) in enumerate(ev, start=1):
            mark = " **[GOLD]**" if cid in gold else ""
            lines.append(f"\n**[{i}]** `{cid}`{mark} (score {score})\n\n{text}\n")
        lines.append("\n### Verdict\n\n- verdict: \n- contradicted sentence: \n"
                     "- notes: \n")

        rows.append({
            "question_id": r["question_id"],
            "question_type": r.get("question_type"),
            "trigger": r.get("correction_reason"),
            "gold_in_evidence": r.get("gold_in_evidence"),
            "verdict": "",
            "contradicted_sentence": "",
            "notes": "",
        })

    out_md.write_text("".join(lines), encoding="utf-8")
    out_csv = RESULTS / f"refusal_audit_{tag}_labels.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_md}\nWrote {out_csv}  ({len(rows)} rows to label)")


if __name__ == "__main__":
    main()