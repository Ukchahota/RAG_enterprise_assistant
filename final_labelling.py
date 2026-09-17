"""
Final correctness labelling - one script, three phases, resumable.

Put this file in the repo root and run:
    python final_labelling.py

PHASE 0  Build (automatic, once)
         Selects the items, writes a hidden key, and pauses so you can commit.
PHASE 1  Key-fact checklists (answers hidden)
         For each question, list the 1-3 facts an answer MUST state.
         Pauses when finished so you can commit the checklists.
PHASE 2  Blind labelling (pooled, shuffled, citation markers removed)
         S0, S1, S2 and S3 answers on the 29-question sample, plus all 45
         withheld S3 drafts. For each fact: present / missing / contradicted.
PHASE 3  Analysis (automatic)
         Writes results/final_labelling/labels.csv and summary.txt.

Label rule (fixed in advance):
    refusal            -> refused
    any fact contradicted -> incorrect
    all facts present  -> correct
    some facts present -> partial      (scores 0 in the primary analysis)
    no facts present   -> incorrect
Empty answers are recorded as 'empty' and excluded from paired tests
(sensitivity analysis counts them as not correct).

Quit any time with q (or Ctrl+C); rerun to resume.
"""

import os
import random
import re
import shutil
import sys
import textwrap
from math import comb, sqrt
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ------------------------------------------------------------------ settings
INCLUDE_S2 = True          # set False to skip S2 answers
SEED = 702
ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = RES / "final_labelling"

ANSWER_FILES = {
    "S0": RES / "answers_s0_llm_only.csv",
    "S1": RES / "answers_s1_basic_rag.csv",
    "S2": RES / "answers_s2_modular_rag.csv",
    "S3": RES / "answers_s3_corrective_rag_clean.csv",
}
SAMPLE_FILES = [RES / "correctness_sample.csv", RES / "correctness_scores.csv"]
QUESTION_FILE = ROOT / "data" / "evaluation" / "test_questions_v2_answerable.csv"
DRAFT_VERIFY_FILE = RES / "hallucination_evaluation_s3_drafts.csv"

KEY = OUT / "items_key.csv"          # hidden: which system / draft each item is
SHEET = OUT / "items_sheet.csv"      # what you see
AUTO = OUT / "auto_labels.csv"       # refused / empty, recorded without showing
FACTS = OUT / "keyfacts.csv"
JUDGE = OUT / "judgements.csv"
LABELS = OUT / "labels.csv"
SUMMARY = OUT / "summary.txt"
BUILT_FLAG = OUT / ".built_committed"
FACTS_FLAG = OUT / ".facts_committed"

LABEL_ORDER = ["correct", "partial", "incorrect", "refused", "empty"]
CITATION = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")


# ------------------------------------------------------------------ helpers
def read(p):
    d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "question_id" in d.columns:
        d["question_id"] = d["question_id"].str.strip()
    return d


def write(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    while True:
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            input(f"{p.name} is locked (Excel or OneDrive). Close it, then press Enter...")


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def show(label, text, limit=None):
    text = str(text)
    if limit and len(text) > limit:
        text = text[:limit] + " ..."
    width = max(60, shutil.get_terminal_size().columns - 4)
    print(label)
    for para in text.splitlines():
        print(textwrap.fill(para, width, initial_indent="  ", subsequent_indent="  ")
              if para.strip() else "")
    print()


def strip_citations(text):
    return re.sub(r"[ \t]+", " ", CITATION.sub("", str(text))).strip()


def pause(msg):
    try:
        input(msg)
    except EOFError:
        sys.exit(0)


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        sys.exit(0)


# ------------------------------------------------------------------ phase 0
def sample_ids():
    for p in SAMPLE_FILES:
        if p.exists():
            ids = read(p)["question_id"].drop_duplicates().tolist()
            if ids:
                return ids, p.name
    sys.exit("No sample file found (correctness_sample.csv or correctness_scores.csv).")


def build():
    for s, p in ANSWER_FILES.items():
        if not p.exists():
            sys.exit(f"Missing {p}")
    ids, src = sample_ids()
    answers = {s: read(p).drop_duplicates("question_id").set_index("question_id")
               for s, p in ANSWER_FILES.items()}
    s3 = answers["S3"]
    action = s3["corrective_action"].str.strip().str.lower()

    items, auto = [], []
    systems = ["S0", "S1", "S2"] if INCLUDE_S2 else ["S0", "S1"]
    for q in ids:
        for s in systems:
            a = answers[s]
            text = a.at[q, "answer"] if q in a.index else ""
            if not str(text).strip():
                auto.append({"question_id": q, "system": s, "kind": "answer", "label": "empty"})
            else:
                items.append({"question_id": q, "system": s, "kind": "answer", "text": text})
        if q not in s3.index or not str(s3.at[q, "answer"]).strip():
            auto.append({"question_id": q, "system": "S3", "kind": "answer", "label": "empty"})
        elif action[q] == "refused":
            auto.append({"question_id": q, "system": "S3", "kind": "answer", "label": "refused"})
        else:
            items.append({"question_id": q, "system": "S3", "kind": "answer",
                          "text": s3.at[q, "answer"]})

    withheld = s3[action == "refused"]
    for q, row in withheld.iterrows():
        if str(row["draft_answer"]).strip():
            items.append({"question_id": q, "system": "S3", "kind": "draft",
                          "text": row["draft_answer"]})
        else:
            auto.append({"question_id": q, "system": "S3", "kind": "draft", "label": "empty"})

    rng = random.Random(SEED)
    rng.shuffle(items)
    key = pd.DataFrame(items)
    key.insert(0, "item_id", [f"FL{i:03d}" for i in range(1, len(key) + 1)])
    sheet = key[["item_id", "question_id"]].copy()
    sheet["answer_shown"] = key["text"].map(strip_citations)

    OUT.mkdir(parents=True, exist_ok=True)
    write(key.drop(columns=["text"]), KEY)
    write(sheet, SHEET)
    write(pd.DataFrame(auto, columns=["question_id", "system", "kind", "label"]), AUTO)
    write(pd.DataFrame(columns=["question_id", "fact_no", "fact"]), FACTS)
    write(pd.DataFrame(columns=["item_id", "fact_no", "status", "note"]), JUDGE)

    qs = sorted(set(key["question_id"]))
    print(f"Sample source: {src} ({len(ids)} questions)")
    print(f"Items to label: {len(key)}  ->  {key.groupby(['system', 'kind']).size().to_dict()}")
    print(f"Recorded automatically (not shown): {pd.DataFrame(auto).groupby(['system', 'label']).size().to_dict() if auto else {}}")
    print(f"Questions needing a key-fact checklist: {len(qs)}")
    print("\nCommit the build before starting (the key file stays hidden - do not open it):")
    print('  git add -f results\\final_labelling final_labelling.py')
    print('  git commit -m "Final labelling: item selection and hidden key, before labelling"')
    BUILT_FLAG.write_text("built", encoding="utf-8")
    pause("\nPress Enter to start Phase 1 (or Ctrl+C to stop and commit first)...")


# ------------------------------------------------------------------ phase 1
def question_info():
    info = {}
    if QUESTION_FILE.exists():
        q = read(QUESTION_FILE)
        for _, r in q.iterrows():
            info[r["question_id"]] = {
                "question": r.get("question", ""),
                "type": r.get("question_type", ""),
                "expected": r.get("expected_answer", ""),
                "gold": r.get("gold_evidence", ""),
            }
    for p in ANSWER_FILES.values():
        a = read(p)
        for _, r in a.iterrows():
            d = info.setdefault(r["question_id"], {})
            d.setdefault("question", r.get("question", ""))
            d.setdefault("type", r.get("question_type", ""))
            d.setdefault("expected", r.get("expected_answer", ""))
            d.setdefault("gold", "")
    return info


def phase1(info):
    key = read(KEY)
    facts = read(FACTS)
    needed = sorted(set(key["question_id"]))
    done = set(facts["question_id"])
    todo = [q for q in needed if q not in done]
    for q in todo:
        d = info.get(q, {})
        while True:
            clear()
            print(f"PHASE 1 - key facts   [{len(done)}/{len(needed)} questions done]   {q}   ({d.get('type', '')})\n")
            show("QUESTION:", d.get("question", ""))
            show("EXPECTED ANSWER:", d.get("expected", ""))
            if d.get("gold"):
                show("GOLD EVIDENCE (shortened):", d["gold"], limit=1200)
            print("List only the facts an answer MUST state to answer the question as asked (1-3).")
            print("One per line; empty line to finish; q to quit.\n")
            items = []
            if d.get("type") == "adversarial_false_premise":
                yn = ask("False-premise question. Add 'Rejects or corrects the false premise' as fact 1? [Y/n] ").strip().lower()
                if yn in ("", "y", "yes"):
                    items.append("Rejects or corrects the false premise in the question")
            while True:
                t = ask(f"Key fact {len(items) + 1}: ").strip()
                if t.lower() == "q":
                    return False
                if not t:
                    if items:
                        break
                    print("  At least one fact is required.")
                    continue
                items.append(t)
            print("\nFacts:")
            for n, t in enumerate(items, 1):
                print(f"  {n}. {t}")
            if ask("\n[k]eep / [r]edo > ").strip().lower() == "k":
                break
        new = pd.DataFrame({"question_id": q, "fact_no": [str(i) for i in range(1, len(items) + 1)],
                            "fact": items})
        facts = pd.concat([facts, new], ignore_index=True)
        write(facts, FACTS)
        done.add(q)
    return True


# ------------------------------------------------------------------ phase 2
def phase2(info):
    sheet = read(SHEET)
    facts = read(FACTS)
    judg = read(JUDGE)
    fact_map = {q: g.sort_values("fact_no", key=lambda s: s.astype(int))
                for q, g in facts.groupby("question_id")}
    history = []
    force = None

    while True:
        judged = set(zip(judg["item_id"], judg["fact_no"]))
        pending = []
        for _, r in sheet.iterrows():
            for fn in fact_map[r["question_id"]]["fact_no"]:
                if (r["item_id"], fn) not in judged and (r["item_id"], "ALL") not in judged:
                    pending.append((r["item_id"], fn))
        if not pending:
            return True
        item, fn = force or pending[0]
        force = None

        row = sheet[sheet["item_id"] == item].iloc[0]
        q = row["question_id"]
        d = info.get(q, {})
        qf = fact_map[q]
        n_items_done = sheet["item_id"].isin(
            [i for i in sheet["item_id"]
             if all((i, f) in judged or (i, "ALL") in judged
                    for f in fact_map[sheet.loc[sheet["item_id"] == i, "question_id"].iloc[0]]["fact_no"])]
        ).sum()
        first_fact = fn == qf["fact_no"].iloc[0]

        clear()
        print(f"PHASE 2 - blind labelling   [{n_items_done}/{len(sheet)} answers done]   item {item}\n")
        show("QUESTION:", d.get("question", ""))
        show("ANSWER (citation markers removed):", row["answer_shown"])
        print("Key facts:")
        for _, fr in qf.iterrows():
            m = judg[(judg["item_id"] == item) & (judg["fact_no"] == fr["fact_no"])]
            tag = m["status"].iloc[0] if len(m) else ("<- now" if fr["fact_no"] == fn else "")
            print(f"  {fr['fact_no']}. {fr['fact']}   {tag}")
        fact = qf.loc[qf["fact_no"] == fn, "fact"].iloc[0]
        print(f"\nFACT {fn}: {fact}")
        print("Judge the answer only. Ignore whether it cites anything.\n")

        opts = "[p]resent  [m]issing  [c]ontradicted"
        if first_fact:
            opts += "  |  [r] whole answer is a refusal"
        choice = ask(f"{opts}  |  [e] show expected answer  [u]ndo  [q]uit > ").strip().lower()

        if choice == "q":
            return False
        if choice == "e":
            show("EXPECTED ANSWER:", d.get("expected", ""))
            pause("Press Enter to continue...")
            force = (item, fn)
            continue
        if choice == "u":
            if history:
                last_item, last_facts = history.pop()
                mask = (judg["item_id"] == last_item) & (judg["fact_no"].isin(last_facts))
                judg = judg[~mask].reset_index(drop=True)
                write(judg, JUDGE)
                force = (last_item, [f for f in last_facts if f != "ALL"][0]
                         if any(f != "ALL" for f in last_facts) else qf["fact_no"].iloc[0])
            else:
                pause("Nothing to undo - press Enter")
            continue
        if choice == "r" and first_fact:
            judg = pd.concat([judg, pd.DataFrame([{"item_id": item, "fact_no": "ALL",
                                                   "status": "refusal", "note": ""}])],
                             ignore_index=True)
            history.append((item, ["ALL"]))
        elif choice in ("p", "m", "c"):
            status = {"p": "present", "m": "missing", "c": "contradicted"}[choice]
            note = ""
            if choice == "c":
                while not note:
                    note = ask("What does the answer say instead? (required): ").strip()
            judg = pd.concat([judg, pd.DataFrame([{"item_id": item, "fact_no": fn,
                                                   "status": status, "note": note}])],
                             ignore_index=True)
            history.append((item, [fn]))
        else:
            pause("Unrecognised key - press Enter")
            continue
        write(judg, JUDGE)


# ------------------------------------------------------------------ phase 3
def derive(statuses):
    s = list(statuses)
    if "refusal" in s:
        return "refused"
    if "contradicted" in s:
        return "incorrect"
    if s and all(x == "present" for x in s):
        return "correct"
    if any(x == "present" for x in s):
        return "partial"
    return "incorrect"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def phase3(info):
    key = read(KEY)
    judg = read(JUDGE)
    auto = read(AUTO)
    facts = read(FACTS)
    nfacts = facts.groupby("question_id").size().to_dict()

    rows = []
    for _, k in key.iterrows():
        j = judg[judg["item_id"] == k["item_id"]]
        rows.append({
            "item_id": k["item_id"], "question_id": k["question_id"], "system": k["system"],
            "kind": k["kind"], "question_type": info.get(k["question_id"], {}).get("type", ""),
            "label": derive(j["status"]), "n_facts": nfacts.get(k["question_id"], 0),
            "n_present": int((j["status"] == "present").sum()),
            "n_contradicted": int((j["status"] == "contradicted").sum()),
            "source": "labelled",
        })
    for _, a in auto.iterrows():
        rows.append({"item_id": "", "question_id": a["question_id"], "system": a["system"],
                     "kind": a["kind"], "question_type": info.get(a["question_id"], {}).get("type", ""),
                     "label": a["label"], "n_facts": nfacts.get(a["question_id"], 0),
                     "n_present": 0, "n_contradicted": 0, "source": "automatic"})
    labels = pd.DataFrame(rows)
    write(labels, LABELS)

    ids, _ = sample_ids()
    ans = labels[labels["kind"] == "answer"]
    drafts = labels[labels["kind"] == "draft"].set_index("question_id")
    per = {s: ans[ans["system"] == s].set_index("question_id")["label"] for s in ["S0", "S1", "S2", "S3"]}
    table = pd.DataFrame({s: per[s].reindex(ids) for s in per if len(per[s])})
    s3 = table["S3"]
    table["S3_ungated"] = [drafts["label"].get(q, "empty") if s3[q] == "refused" else s3[q] for q in ids]

    L = []
    L.append("FINAL CORRECTNESS LABELLING - SUMMARY")
    L.append(f"Sample: {len(ids)} questions; withheld S3 drafts labelled: {len(drafts)}")
    L.append("Primary rule: strict (partial = 0); empty answers excluded from paired tests.\n")

    L.append("1. LABEL COUNTS ON THE SAMPLE")
    L.append(pd.DataFrame({c: table[c].value_counts().reindex(LABEL_ORDER, fill_value=0)
                           for c in table.columns}).to_string())

    L.append("\n2. ACCURACY (non-empty answers; 95% Wilson intervals for strict)")
    for c in table.columns:
        v = table[c][table[c] != "empty"]
        n = len(v)
        k = int((v == "correct").sum())
        half = k + 0.5 * int((v == "partial").sum())
        lenient = k + int((v == "partial").sum())
        lo, hi = wilson(k, n)
        L.append(f"  {c:<11} strict {k}/{n} = {k / n if n else float('nan'):.3f} [{lo:.3f}, {hi:.3f}]"
                 f"   half {half / n if n else float('nan'):.3f}   lenient {lenient / n if n else float('nan'):.3f}")

    pairs = [("S0", "S1"), ("S1", "S2"), ("S0", "S3"), ("S1", "S3"), ("S2", "S3"), ("S3", "S3_ungated")]
    pairs = [p for p in pairs if p[0] in table and p[1] in table]
    for num, (title, empty_as_wrong) in zip(["3a", "3b"],
                                            [("primary: empty answers excluded", False),
                                             ("sensitivity: empty answers count as not correct", True)]):
        L.append(f"\n{num}. PAIRED EXACT McNEMAR, strict ({title}; Holm-adjusted)")
        res = []
        for a, b in pairs:
            both = table[[a, b]]
            if not empty_as_wrong:
                both = both[(both[a] != "empty") & (both[b] != "empty")]
            ca, cb = both[a] == "correct", both[b] == "correct"
            only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
            res.append((f"{a} vs {b}", len(both), int((ca & cb).sum()), only_a, only_b,
                        int((~ca & ~cb).sum()), mcnemar(only_a, only_b)))
        adj = holm([r[-1] for r in res])
        L.append(f"  {'pair':<18}{'n':>4}{'both':>6}{'A only':>8}{'B only':>8}{'neither':>9}{'p':>9}{'p Holm':>9}")
        for r, pa in zip(res, adj):
            L.append(f"  {r[0]:<18}{r[1]:>4}{r[2]:>6}{r[3]:>8}{r[4]:>8}{r[5]:>9}{r[6]:>9.4f}{pa:>9.4f}")

    L.append("\n4. WITHHELD S3 DRAFTS (all withheld questions)")
    dl = drafts["label"]
    n = int((dl != "empty").sum())
    k = int((dl == "correct").sum())
    lo, hi = wilson(n - k, n)
    L.append(dl.value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())
    L.append(f"  Correct drafts withheld (cost): {k}/{n}")
    L.append(f"  Gate precision (withheld drafts not fully correct): {n - k}/{n} = "
             f"{(n - k) / n if n else float('nan'):.3f} [{lo:.3f}, {hi:.3f}]")
    L.append(f"  Withheld drafts judged incorrect: {int((dl == 'incorrect').sum())}; partial: {int((dl == 'partial').sum())}")

    s3a = read(ANSWER_FILES["S3"]).set_index("question_id")
    dd = pd.DataFrame({"label": dl})
    num = lambda c: pd.to_numeric(s3a[c], errors="coerce").reindex(dd.index)
    con, rate, gold = num("draft_n_contradicted"), num("draft_unsupported_rate"), num("gold_in_evidence")
    dd["runtime_trigger"] = [
        "both" if (c > 0 and r > 0.4) else "contradiction only" if c > 0 else "rate only" if r > 0.4 else "other"
        for c, r in zip(con.fillna(0), rate.fillna(0))]
    dd["retrieval"] = ["complete" if g >= 1.0 else "incomplete" for g in gold.fillna(0)]
    dd["question_type"] = [info.get(q, {}).get("type", "") for q in dd.index]
    if DRAFT_VERIFY_FILE.exists():
        dv = read(DRAFT_VERIFY_FILE).drop_duplicates("question_id").set_index("question_id")
        nu = pd.to_numeric(dv["n_unsupported"], errors="coerce").reindex(dd.index)
        dd["current_verifier"] = ["no unsupported claim" if x == 0 else "has unsupported claim" if x > 0 else "n/a"
                                  for x in nu]
    for col in ["runtime_trigger", "retrieval", "current_verifier", "question_type"]:
        if col in dd:
            L.append(f"\n  Draft labels by {col}:")
            L.append(pd.crosstab(dd[col], dd["label"]).reindex(columns=[c for c in LABEL_ORDER if c in set(dd["label"])],
                                                                 fill_value=0).to_string())

    L.append("\n5. S3 GATED vs UNGATED ON THE SAMPLE")
    for c in ["S3", "S3_ungated"]:
        v = table[c][table[c] != "empty"]
        L.append(f"  {c:<11} correct {int((v == 'correct').sum())}/{len(v)}")
    delivered = table["S3"][~table["S3"].isin(["refused", "empty"])]
    L.append(f"  Delivered S3 answers not fully correct (let through by the gate): "
             f"{int((delivered != 'correct').sum())}/{len(delivered)}")

    L.append("\n6. SAMPLE LABELS BY QUESTION")
    L.append(table.assign(type=[info.get(q, {}).get("type", "") for q in table.index]).to_string())

    text = "\n".join(L)
    SUMMARY.write_text(text, encoding="utf-8")
    clear()
    print(text)
    print(f"\nSaved {LABELS.relative_to(ROOT)} and {SUMMARY.relative_to(ROOT)}")
    print("Commit them:  git add -f results\\final_labelling  &&  git commit -m \"Final labelling results\"")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    try:
        if not KEY.exists():
            build()
        info = question_info()
        if not phase1(info):
            print("\nStopped in Phase 1. Rerun: python final_labelling.py")
            sys.exit(0)
        if not FACTS_FLAG.exists():
            clear()
            print("PHASE 1 COMPLETE. Commit the checklists BEFORE labelling any answer:")
            print('  git add -f results\\final_labelling')
            print('  git commit -m "Final labelling: key-fact checklists, before labelling"')
            FACTS_FLAG.write_text("facts", encoding="utf-8")
            pause("\nPress Enter to start Phase 2 (or Ctrl+C to stop and commit first)...")
        if not phase2(info):
            print("\nStopped in Phase 2. Rerun: python final_labelling.py")
            sys.exit(0)
        phase3(info)
    except KeyboardInterrupt:
        print("\nStopped. Everything entered so far is saved. Rerun: python final_labelling.py")