"""
Collect the evidence needed to fill the dissertation's open items.

Put this file in the repo root and run:
    python collect_evidence.py

It only READS project files (and runs a few existing analysis scripts, whose
output it captures). Everything is written to evidence_dump.txt - upload that
file to Claude. Each section is labelled with the open-item number it serves.
"""

import contextlib
import io
import json
import re
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "evidence_dump.txt"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode"}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

buf = io.StringIO()


def say(*a):
    print(*a, file=buf)


def section(title):
    say("\n" + "=" * 90)
    say(title)
    say("=" * 90)


def guarded(fn):
    def run():
        try:
            fn()
        except Exception:
            say(f"!! {fn.__name__} failed:")
            say(traceback.format_exc(limit=3))
    run.__name__ = fn.__name__
    return run


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- A
@guarded
def environment():
    section("A. ENVIRONMENT (items 10, references: software versions)")
    say("python", sys.version.split()[0])
    for pkg in ["pandas", "numpy", "faiss-cpu", "faiss-gpu", "rank-bm25", "rank_bm25",
                "sentence-transformers", "transformers", "torch", "streamlit", "openai",
                "scikit-learn", "pypdf", "PyPDF2", "pdfplumber", "beautifulsoup4", "langchain",
                "langchain-text-splitters", "nltk", "spacy"]:
        try:
            say(f"  {pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass


# ---------------------------------------------------------------- B
@guarded
def layout():
    section("B. REPOSITORY LAYOUT (items 14, 16)")
    for p in sorted(ROOT.iterdir()):
        if p.name in SKIP_DIRS:
            continue
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file() and not (set(f.parts) & SKIP_DIRS)]
            exts = pd.Series([f.suffix.lower() or "(none)" for f in files]).value_counts()
            say(f"[dir] {p.name}/  {len(files)} files  " + ", ".join(f"{k}:{v}" for k, v in exts.head(8).items()))
            if p.name in {"src", "app", "tests", "test"}:
                for f in sorted(p.rglob("*.py")):
                    if not (set(f.parts) & SKIP_DIRS):
                        say(f"      {rel(f)}  ({sum(1 for _ in open(f, encoding='utf-8', errors='replace'))} lines)")
        else:
            say(f"      {p.name}")
    for name in ["data", "corpus", "docs", "documents", "raw", "raw_docs", "policies"]:
        d = ROOT / name
        if d.is_dir():
            say(f"\nFiles in {name}/ (first 40):")
            for f in sorted(x for x in d.rglob("*") if x.is_file())[:40]:
                say(f"  {rel(f)}  {f.stat().st_size // 1024} KB")


# ---------------------------------------------------------------- C
@guarded
def threshold_sites():
    section("C. RETRIEVAL-COMPLETENESS THRESHOLD SITES (item 2) - should use >= 1.0")
    for fname, line in [("analyse_failure_layers.py", 20), ("attribution_agreement.py", 7),
                        ("src/evaluate_hallucination.py", 214)]:
        p = ROOT / fname
        if not p.exists():
            say(f"{fname}: NOT FOUND")
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        say(f"\n{fname} (lines {max(1, line - 3)}-{line + 3}):")
        for k in range(max(0, line - 4), min(len(lines), line + 3)):
            say(f"  {k + 1:4d}: {lines[k]}")
        hits = [f"{i + 1}: {l.strip()}" for i, l in enumerate(lines) if "gold_in_evidence" in l]
        say("  all lines mentioning gold_in_evidence:")
        for h in hits[:15]:
            say("   ", h)


# ---------------------------------------------------------------- D
PATTERN = re.compile(
    r"(model|MODEL|SentenceTransformer|CrossEncoder|faiss|FAISS|BM25|bm25|top_k|TOP_K|\bk\s*=|"
    r"threshold|THRESHOLD|temperature|max_tokens|chunk_size|CHUNK|overlap|OVERLAP|0\.4\b|"
    r"refus|REFUS|NON_CLAIM|entail|contradict|supported_uncited|nei|citation|repair|"
    r"rewrite|hybrid|metadata|st\.tabs|sent_tokenize|split|PdfReader|pdfplumber|BeautifulSoup|"
    r"refusal_accuracy|faithfulness\s*=|unsupported_claim_rate|citation_accuracy|recall)"
)


@guarded
def config_grep():
    section("D. CONFIGURATION AND LOGIC LINES IN src/ AND app/ (items 10, 11, 13, 16, 21)")
    files = [f for d in ["src", "app"] if (ROOT / d).is_dir()
             for f in sorted((ROOT / d).rglob("*.py")) if not (set(f.parts) & SKIP_DIRS)]
    for f in files:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [(i + 1, l.rstrip()) for i, l in enumerate(lines)
                if PATTERN.search(l) and not l.strip().startswith("#") and len(l.strip()) > 3]
        if not hits:
            continue
        say(f"\n--- {rel(f)} ({len(hits)} matching lines, showing up to 80)")
        for n, l in hits[:80]:
            say(f"  {n:4d}: {l[:160]}")


@guarded
def prompts():
    section("D2. PROMPT TEXT (item 10) - long string literals mentioning the assistant's task")
    files = [f for d in ["src", "app"] if (ROOT / d).is_dir()
             for f in sorted((ROOT / d).rglob("*.py")) if not (set(f.parts) & SKIP_DIRS)]
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'("""|\'\'\')(.*?)(\1)', text, flags=re.S):
            body = m.group(2)
            if re.search(r"(?i)(you are|answer|context|cite|citation|passage)", body) and len(body) > 80:
                line = text[:m.start()].count("\n") + 1
                say(f"\n--- {rel(f)} line {line}:\n{body.strip()[:1500]}")


# ---------------------------------------------------------------- E
SYSTEM_GLOBS = {
    "s0": ["answers_s0*.csv"],
    "s1": ["answers_s1*.csv"],
    "s2": ["answers_s2*.csv"],
    "s3": ["answers_s3_corrective_rag_clean.csv", "answers_s3*.csv"],
}


def pick_answer_file(system):
    for pattern in SYSTEM_GLOBS[system]:
        cands = sorted(RES.glob(pattern), key=lambda p: (("backup" in p.name) or ("drafts" in p.name), len(p.name)))
        cands = [c for c in cands if "backup" not in c.name and "drafts" not in c.name]
        if cands:
            return cands[0]
    return None


ANSWERS = {}
COMMON = set()


@guarded
def questions_and_common_set():
    section("E. QUESTION SETS, QUESTION TYPES AND THE COMMON SET (items 18, 19)")
    for f in sorted(ROOT.rglob("*")):
        if (f.is_file() and f.suffix.lower() in {".csv", ".json", ".jsonl"}
                and re.search(r"(?i)question|eval_set|gold", f.name)
                and not (set(f.parts) & SKIP_DIRS) and "results" not in f.parts):
            say(f"\nCandidate question file: {rel(f)}")
            try:
                if f.suffix.lower() == ".csv":
                    q = read_csv(f)
                elif f.suffix.lower() == ".jsonl":
                    q = pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()])
                else:
                    data = json.load(open(f, encoding="utf-8"))
                    q = pd.DataFrame(data if isinstance(data, list) else data.get("questions", []))
                say(f"  rows {len(q)}; columns {list(q.columns)[:15]}")
                if "question_type" in q.columns:
                    say(q["question_type"].value_counts().to_string())
            except Exception as e:
                say(f"  could not read: {e}")

    for s in ["s0", "s1", "s2", "s3"]:
        p = pick_answer_file(s)
        others = [x.name for x in RES.glob(SYSTEM_GLOBS[s][-1])]
        say(f"\n{s.upper()} answer file used: {p.name if p else 'NONE'}   (all matches: {others})")
        if p is None:
            continue
        df = read_csv(p)
        df["question_id"] = df["question_id"].str.strip()
        ANSWERS[s] = df
        empty = df["answer"].str.strip().eq("")
        err = df["error"].str.strip().ne("") if "error" in df.columns else pd.Series(False, index=df.index)
        say(f"  rows {len(df)}, unique ids {df['question_id'].nunique()}, empty answers {int(empty.sum())}, errors {int(err.sum())}")
        say(f"  empty/error ids: {sorted(df.loc[empty | err, 'question_id'])}")

    if len(ANSWERS) == 4:
        ok = [set(d.loc[d["answer"].str.strip().ne(""), "question_id"]) for d in ANSWERS.values()]
        COMMON.update(set.intersection(*ok))
        say(f"\nCommon set (non-empty answer in all four systems): {len(COMMON)} questions")
        s0 = ANSWERS["s0"].drop_duplicates("question_id").set_index("question_id")
        if "question_type" in s0.columns:
            tab = pd.DataFrame({
                "all": s0["question_type"].value_counts(),
                "common": s0.loc[sorted(COMMON), "question_type"].value_counts(),
            }).fillna(0).astype(int)
            tab.loc["TOTAL"] = tab.sum()
            say(tab.to_string())
        if "looks_like_refusal" in s0.columns:
            say(f"\nS0 looks_like_refusal counts: {s0['looks_like_refusal'].value_counts().to_dict()}")


# ---------------------------------------------------------------- F
@guarded
def s3_refusals():
    section("F. S3 WITHHOLDING: COUNTS, TRIGGERS, RETRIEVAL COMPLETENESS (items 3, 8, 11, 12)")
    s3 = ANSWERS.get("s3")
    if s3 is None:
        say("no S3 file")
        return
    say(f"corrective_action counts (all rows): {s3['corrective_action'].value_counts().to_dict()}")
    for label, d in [("ALL ROWS", s3), ("COMMON SET", s3[s3["question_id"].isin(COMMON)])]:
        act = d["corrective_action"].str.strip().str.lower()
        withheld = d[~act.isin(["accepted", "accept", "none", ""])]
        rate = num(withheld["draft_unsupported_rate"])
        con = num(withheld["draft_n_contradicted"])
        gold = num(withheld["gold_in_evidence"])
        say(f"\n{label}: n={len(d)}, withheld={len(withheld)}  (actions: {act.value_counts().to_dict()})")
        say(f"  trigger: contradiction only {int(((con > 0) & ~(rate > 0.4)).sum())}, "
            f"rate only {int((~(con > 0) & (rate > 0.4)).sum())}, both {int(((con > 0) & (rate > 0.4)).sum())}, "
            f"neither {int((~(con > 0) & ~(rate > 0.4)).sum())}")
        say(f"  gold_in_evidence >= 1.0: {int((gold >= 1.0).sum())}; > 0: {int((gold > 0).sum())}; missing: {int(gold.isna().sum())}")
        checkable = num(withheld["draft_n_supported"]).fillna(0) + num(withheld["draft_n_unsupported"]).fillna(0) + con.fillna(0)
        say(f"  sup+uns+con claims per withheld draft: {checkable.describe().round(2).to_dict()}")
        say(f"  correction_reason starts: {withheld['correction_reason'].str[:40].value_counts().head(6).to_dict()}")

    say("\nn_withheld (claim-level) summary by action:")
    say(s3.assign(nw=num(s3["n_withheld"])).groupby("corrective_action")["nw"].agg(["count", "sum", "mean"]).to_string())
    acc = s3[s3["corrective_action"].str.lower().eq("accepted")]
    same = (acc["answer"].str.split().str.join(" ") == acc["draft_answer"].str.split().str.join(" "))
    say(f"Accepted answers identical to their draft: {int(same.sum())} of {len(acc)}")

    s2 = ANSWERS.get("s2")
    if s2 is not None:
        m = s3.merge(s2[["question_id", "answer"]].rename(columns={"answer": "s2_answer"}), on="question_id")
        same = (m["draft_answer"].str.split().str.join(" ") == m["s2_answer"].str.split().str.join(" "))
        say(f"S3 draft identical to S2 answer: {int(same.sum())} of {len(m)}")
        m2 = m[m["question_id"].isin(COMMON)]
        say(f"  on common set: {int(same[m['question_id'].isin(COMMON)].sum())} of {len(m2)}")


# ---------------------------------------------------------------- G
@guarded
def retrieval_files():
    section("G. RETRIEVAL EVALUATION FILES (items 6, 13)")
    for f in sorted(RES.glob("*retriev*.csv")):
        d = read_csv(f)
        say(f"\n{f.name}: rows {len(d)}")
        metrics = [c for c in d.columns if re.search(r"(hit|mrr|ndcg|precision|recall|coverage|top_1)", c)]
        means = {c: round(num(d[c]).mean(), 3) for c in metrics if num(d[c]).notna().any()}
        say(f"  means: {means}")
        for c in ["correction_applied", "rewrite_applied", "quality_passed", "retrieval_stage"]:
            if c in d.columns:
                say(f"  {c}: {d[c].value_counts().head(6).to_dict()}")
        for c in ["system", "retriever", "method", "config"]:
            if c in d.columns:
                say(f"  {c}: {d[c].value_counts().to_dict()}")


# ---------------------------------------------------------------- H
@guarded
def claim_level_summaries():
    section("H. CLAIM-LEVEL SUMMARIES PER SYSTEM (items 5, 7, 21, Table 8)")
    cols = ["faithfulness", "unsupported_claim_rate", "citation_accuracy", "n_checkable",
            "n_supported", "n_supported_uncited", "n_unsupported", "n_contradicted", "n_nei", "gold_in_evidence"]
    for f in sorted(RES.glob("hallucination_evaluation*.csv")):
        d = read_csv(f)
        d["question_id"] = d["question_id"].str.strip()
        say(f"\n{f.name}: rows {len(d)}; systems {d['system'].value_counts().to_dict() if 'system' in d else '-'}")
        present = [c for c in cols if c in d.columns]
        for label, sub in [("all", d), ("common", d[d["question_id"].isin(COMMON)])]:
            if "system" in sub.columns and len(sub):
                g = sub.assign(**{c: num(sub[c]) for c in present}).groupby("system")[present].mean().round(3)
                say(f"  [{label}]\n{g.to_string()}")
    s3 = ANSWERS.get("s3")
    if s3 is not None:
        c = s3[s3["question_id"].isin(COMMON)]
        say(f"\nS3 draft means on common set: faithfulness {num(c['draft_faithfulness']).mean():.3f}, "
            f"unsupported {num(c['draft_unsupported_rate']).mean():.3f}, citation acc {num(c['draft_citation_accuracy']).mean():.3f}")


@guarded
def claims_files():
    section("I. CLAIM FILES: LABEL COUNTS AND THE DILUTION POPULATION (item 4, section 4.4)")
    for f in sorted(RES.glob("hallucination_claims*.csv")):
        d = read_csv(f)
        say(f"\n{f.name}: rows {len(d)}")
        if {"system", "label"} <= set(d.columns):
            say(pd.crosstab(d["system"], d["label"]).to_string())
        if {"label", "best_entailment"} <= set(d.columns):
            u = d[d["label"] == "unsupported"]
            e = num(u["best_entailment"])
            say(f"  unsupported claims: {len(u)}; best_entailment in [0.2,0.5): {int(((e >= 0.2) & (e < 0.5)).sum())}; "
                f"< 0.2: {int((e < 0.2).sum())}; >= 0.5: {int((e >= 0.5).sum())}")
        if "sentence" in d.columns:
            say(f"  total claims: {len(d)}")


# ---------------------------------------------------------------- J
@guarded
def detector_validation():
    section("J. DETECTOR VALIDATION SHEETS (items 4, 23)")
    for f in sorted(RES.glob("detector_validation*.csv")):
        d = read_csv(f)
        say(f"\n{f.name}: rows {len(d)}; columns {list(d.columns)}")
        for c in d.columns:
            if re.search(r"(?i)(label|human|pred|verdict|gold)", c):
                say(f"  {c}: {d[c].value_counts().to_dict()}")
        pred = next((c for c in d.columns if c in {"label", "predicted_label", "detector_label", "pred_label"}), None)
        human = next((c for c in d.columns if re.search(r"(?i)human", c)), None)
        if pred and human:
            say(pd.crosstab(d[pred], d[human], margins=True).to_string())


# ---------------------------------------------------------------- K
@guarded
def chunks():
    section("K. CHUNK STORE (item 16, section 3.2)")
    cands = [f for f in ROOT.rglob("*") if f.is_file() and not (set(f.parts) & SKIP_DIRS)
             and re.search(r"(?i)chunk", f.name) and f.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet", ".pkl", ".pickle"}]
    for f in sorted(cands)[:6]:
        say(f"\n{rel(f)} ({f.stat().st_size // 1024} KB)")
        if f.suffix.lower() in {".pkl", ".pickle"}:
            say("  (pickle - not opened)")
            continue
        try:
            if f.suffix.lower() == ".csv":
                d = read_csv(f)
            elif f.suffix.lower() == ".parquet":
                d = pd.read_parquet(f)
            elif f.suffix.lower() == ".jsonl":
                d = pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()])
            else:
                data = json.load(open(f, encoding="utf-8"))
                d = pd.json_normalize(data if isinstance(data, list) else data.get("chunks", data))
            say(f"  rows {len(d)}; columns {list(d.columns)[:20]}")
            text_col = next((c for c in d.columns if c.lower() in {"text", "chunk", "content", "chunk_text", "page_content"}), None)
            if text_col:
                L = d[text_col].astype(str).str.len()
                say(f"  length: {L.describe().round(1).to_dict()}; under 50 chars: {int((L < 50).sum())}; under 100: {int((L < 100).sum())}")
            for c in d.columns:
                if re.search(r"(?i)(source|doc|file|title)", c) and d[c].nunique() <= 40:
                    say(f"  {c} ({d[c].nunique()} values): {d[c].value_counts().to_dict()}")
                    break
            say(f"  first row: {d.iloc[0].astype(str).str[:80].to_dict()}")
        except Exception as e:
            say(f"  could not read: {e}")


# ---------------------------------------------------------------- L
@guarded
def search_scripts():
    section("L. WHERE KEY FIGURES ARE COMPUTED (items 5, 20, 22)")
    pats = ["refusal_accuracy", "refusal accuracy", "over_answer", "recall", "0.788", "failed", "is_failure",
            "failure", "precision"]
    files = [f for f in list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")) if f.name != Path(__file__).name]
    for f in sorted(files):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines) if any(p in l for p in pats) and not l.strip().startswith("#")]
        if hits:
            say(f"\n--- {rel(f)}")
            for n, l in hits[:25]:
                say(f"  {n:4d}: {l[:150]}")


@guarded
def run_scripts():
    section("M. OUTPUT OF EXISTING ANALYSIS SCRIPTS (items 2, 14, 15, 23)")
    say("NOTE: failure-layer and attribution outputs are only valid once section C shows >= 1.0 at all three sites.")
    for name in ["check_project.py", "score_detector.py", "analyse_failure_layers.py",
                 "attribution_agreement.py", "analyse_latency.py"]:
        p = ROOT / name
        if not p.exists():
            say(f"\n--- {name}: NOT FOUND")
            continue
        try:
            r = subprocess.run([sys.executable, str(p)], cwd=ROOT, capture_output=True, text=True,
                               timeout=300, encoding="utf-8", errors="replace")
            outtxt = (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")
            say(f"\n--- {name} (exit {r.returncode})")
            say(outtxt[:7000] + ("\n...[truncated]" if len(outtxt) > 7000 else ""))
        except subprocess.TimeoutExpired:
            say(f"\n--- {name}: timed out after 300 s")


if __name__ == "__main__":
    steps = [environment, layout, threshold_sites, config_grep, prompts, questions_and_common_set,
             s3_refusals, retrieval_files, claim_level_summaries, claims_files, detector_validation,
             chunks, search_scripts, run_scripts]
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step.__name__}...", flush=True)
        step()
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nDone. Upload {OUT.name} ({OUT.stat().st_size // 1024} KB) to Claude.")