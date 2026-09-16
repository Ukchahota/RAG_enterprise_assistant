"""
Second evidence pass: the remaining open items in the dissertation draft.

Put this file in the repo root and run:
    python collect_evidence2.py

Reads files only, apart from running sweep_trigger.py. Writes evidence_dump2.txt;
upload that file to Claude. Section numbers match the draft's open items.
"""

import io
import re
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "evidence_dump2.txt"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
buf = io.StringIO()


def say(*a):
    print(*a, file=buf)


def section(title):
    say("\n" + "=" * 90 + "\n" + title + "\n" + "=" * 90)


def guarded(fn):
    def run():
        try:
            fn()
        except Exception:
            say(f"!! {fn.__name__} failed:\n{traceback.format_exc(limit=3)}")
    run.__name__ = fn.__name__
    return run


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def text(p):
    p = ROOT / p
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def show_lines(path, start, end):
    t = text(path)
    if t is None:
        say(f"{path}: NOT FOUND")
        return
    lines = t.splitlines()
    say(f"\n--- {path} lines {start}-{end}")
    for i in range(max(0, start - 1), min(len(lines), end)):
        say(f"  {i + 1:4d}: {lines[i]}")


def show_function(path, name, max_lines=60):
    t = text(path)
    if t is None:
        say(f"{path}: NOT FOUND")
        return
    lines = t.splitlines()
    for i, l in enumerate(lines):
        if re.match(rf"\s*def {name}\b", l):
            say(f"\n--- {path}: def {name} (line {i + 1})")
            indent = len(l) - len(l.lstrip())
            for j in range(i, min(len(lines), i + max_lines)):
                if j > i and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent \
                        and not lines[j].lstrip().startswith((")", "]", "}")):
                    break
                say(f"  {j + 1:4d}: {lines[j]}")
            return
    say(f"{path}: def {name} not found")


def grep(path, pattern, limit=30):
    t = text(path)
    if t is None:
        say(f"{path}: NOT FOUND")
        return
    hits = [(i + 1, l.rstrip()) for i, l in enumerate(t.splitlines()) if re.search(pattern, l)]
    say(f"\n--- {path}: /{pattern}/ ({len(hits)} hits)")
    for n, l in hits[:limit]:
        say(f"  {n:4d}: {l[:170]}")


def as_bool(series):
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0})


@guarded
def item2_model():
    section("ITEM 2. GENERATOR IDENTIFIER IN THE ANSWER FILES")
    for f in sorted(RES.glob("answers_s*.csv")):
        d = read_csv(f)
        if "model" in d.columns:
            say(f"{f.name}: {d['model'].value_counts().to_dict()}")
    t = text(".env")
    if t is not None:
        say(".env keys (values hidden): " + ", ".join(
            l.split("=")[0].strip() for l in t.splitlines() if "=" in l and not l.strip().startswith("#")))
        m = re.search(r"NVIDIA_MODEL\s*=\s*(\S+)", t)
        say(f".env NVIDIA_MODEL = {m.group(1) if m else '(not set)'}")


@guarded
def item3_refusal():
    section("ITEM 3. REFUSAL EVALUATION")
    grep("src/evaluate_refusal.py", r"\.csv|QUESTION|questions_path|read_csv")
    show_lines("src/evaluate_refusal.py", 30, 45)
    p = RES / "refusal_evaluation.csv"
    if p.exists():
        d = read_csv(p)
        say(f"\nrefusal_evaluation.csv: rows {len(d)}; columns {list(d.columns)}")
        if "question_type" in d.columns:
            say(d["question_type"].value_counts().to_string())
        for s in ["S0", "S1", "S2", "S3"]:
            c = f"{s}_refused"
            if c in d.columns:
                b = as_bool(d[c])
                say(f"  {c}: {int(b.sum())}/{b.notna().sum()} = {b.mean():.3f}")
                if "question_type" in d.columns:
                    say("    by type: " + d.assign(b=b).groupby("question_type")["b"].agg(["sum", "count"]).to_dict("index").__str__())
        for c in ["below_threshold", "S3_unsupported_rate"]:
            if c in d.columns:
                say(f"  {c}: {d[c].value_counts().head(8).to_dict()}")
    else:
        say("results/refusal_evaluation.csv NOT FOUND")


@guarded
def item4_retrieval():
    section("ITEM 4. HIT@5 FOR EVERY RETRIEVER, AND HOW MRR IS COMPUTED")
    for name in ["retrieval_evaluation_faiss_v2_FINAL.csv", "retrieval_evaluation_bm25_v2_FINAL.csv",
                 "retrieval_evaluation_hybrid_v2_FINAL.csv", "retrieval_evaluation_query_rewritten_v2_FINAL.csv",
                 "retrieval_evaluation_corrective_v2_FINAL.csv", "retrieval_evaluation_final_reranked.csv"]:
        p = RES / name
        if not p.exists():
            say(f"{name}: NOT FOUND")
            continue
        d = read_csv(p)
        out = {}
        for c in ["top_1_hit", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "full_evidence_coverage_at_5",
                  "full_evidence_at_5", "bm25_hit_at_5"]:
            if c in d.columns:
                b = as_bool(d[c])
                if b.isna().all():
                    b = pd.to_numeric(d[c], errors="coerce")
                out[c] = round(float(b.mean()), 3)
        say(f"{name}: n={len(d)} {out}")
    grep("src/evaluate_final_retriever.py", r"mrr|MRR|CANDIDATE_K|range\(|for k in")
    grep("src/evaluate_s2_modular_rag.py", r"CANDIDATE_K\s*=")


@guarded
def item5_prompts():
    section("ITEM 5. PROMPT BUILDERS IN THE FOUR SYSTEM SCRIPTS")
    for f in ["src/evaluate_s0_llm_only.py", "src/evaluate_s1_basic_rag.py",
              "src/evaluate_s2_modular_rag.py", "src/evaluate_s3_corrective_rag.py"]:
        show_function(f, "build_prompt", 70)
        grep(f, r"^[A-Z_]+\s*=\s*\(?\s*[\"']")
    show_lines("src/evaluate_s3_corrective_rag.py", 36, 60)
    a, b = text("src/evaluate_s2_modular_rag.py"), text("src/evaluate_s3_corrective_rag.py")
    if a and b:
        def body(t):
            m = re.search(r"def build_prompt.*?(?=\ndef |\Z)", t, flags=re.S)
            return m.group(0) if m else ""
        say(f"\nS2 and S3 build_prompt identical: {body(a) == body(b)}")


@guarded
def item6_failure_rules():
    section("ITEM 6. FAILURE-LAYER RULES")
    show_lines("analyse_failure_layers.py", 1, 40)
    show_function("analyse_failure_layers.py", "assign_layer", 40)
    show_lines("analyse_failure_layers.py", 60, 100)
    grep("src/evaluate_hallucination.py", r"GROUNDED_MAX_UNSUPPORTED_RATE|answer_ok|retrieval_ok")


@guarded
def item7_retriever_details():
    section("ITEM 7. REWRITING METHOD, RRF CONSTANT, EMBEDDING NORMALISATION")
    show_function("src/query_rewritten_retriever.py", "rewrite_query", 90)
    grep("src/hybrid_retriever.py", r"rrf_k|candidate_k|def __init__|def retrieve")
    grep("src/evaluate_hybrid_retriever.py", r"candidate_k|rrf_k|HybridRetriever\(")
    grep("src/evaluate_query_rewritten_retrieval.py", r"candidate_k|rrf_k|Retriever\(")
    for f in ["src/embeddings.py", "src/retriever.py", "src/evaluate_s1_basic_rag.py"]:
        grep(f, r"normali[sz]e")


@guarded
def item8_dilution():
    section("ITEM 8. DILUTION AND BOUNDARY TESTS")
    for f in sorted(ROOT.glob("check_dilution*.py")) + [ROOT / "check_boundary_effect.py", ROOT / "check_v2q008.py",
                                                        ROOT / "check_reproduce_v2q008.py"]:
        if f.exists():
            t = f.read_text(encoding="utf-8", errors="replace")
            doc = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', t, flags=re.S)
            outs = re.findall(r"[\w/\\.-]+\.(?:csv|txt|md)", t)
            say(f"\n--- {f.name}: docstring: {(doc.group(1).strip()[:400] if doc else '(none)')}")
            say(f"    files mentioned: {sorted(set(outs))[:10]}")
    for f in sorted(list(RES.glob("*dilution*")) + list(RES.glob("*boundary*")) + list(RES.glob("*v2q008*"))):
        say(f"\n{f.name}")
        if f.suffix == ".csv":
            d = read_csv(f)
            say(f"  rows {len(d)}; columns {list(d.columns)}")
            say(d.head(25).to_string()[:4000])
        else:
            say(f.read_text(encoding="utf-8", errors="replace")[:3000])


@guarded
def item9_sweep():
    section("ITEM 9. TRIGGER SWEEP (sweep_trigger.py)")
    p = ROOT / "sweep_trigger.py"
    if not p.exists():
        say("sweep_trigger.py NOT FOUND")
        return
    show_lines("sweep_trigger.py", 1, 30)
    r = subprocess.run([sys.executable, str(p)], cwd=ROOT, capture_output=True, text=True,
                       timeout=300, encoding="utf-8", errors="replace")
    say(f"\n--- output (exit {r.returncode})\n{(r.stdout or '')[:6000]}")
    if r.stderr.strip():
        say(f"[stderr]\n{r.stderr[:2000]}")
    ans = RES / "answers_s3_corrective_rag_clean.csv"
    if ans.exists():
        d = read_csv(ans)
        ref = d[d["corrective_action"].str.lower() == "refused"].copy()
        n = pd.to_numeric
        sup = n(ref["draft_n_supported"], errors="coerce").fillna(0)
        uns = n(ref["draft_n_unsupported"], errors="coerce").fillna(0)
        con = n(ref["draft_n_contradicted"], errors="coerce").fillna(0)
        unc = n(ref.get("draft_n_supported_uncited", pd.Series(0, index=ref.index)), errors="coerce").fillna(0)
        say(f"\nColumns available for recomputation: {[c for c in d.columns if c.startswith('draft_n')]}")
        checkable_wo_con = sup + unc + uns
        rate_wo_con = (uns / checkable_wo_con.where(checkable_wo_con > 0))
        still = (rate_wo_con > 0.4)
        say(f"Withheld drafts that would STILL be withheld if contradicted claims were ignored entirely "
            f"(rate = unsupported / (checkable - contradicted) > 0.4): {int(still.sum())} of {len(ref)}")
        say("  (if draft_n_supported_uncited is missing, 'checkable' is understated - check the column list above)")


@guarded
def item10_missing():
    section("ITEM 10. S3 QUESTIONS WITHOUT AN ANSWER; LATENCY SET")
    raw, clean = RES / "answers_s3_corrective_rag.csv", RES / "answers_s3_corrective_rag_clean.csv"
    if raw.exists() and clean.exists():
        r, c = read_csv(raw), read_csv(clean)
        missing = r[~r["question_id"].isin(c["question_id"])]
        say(f"raw rows {len(r)}, clean rows {len(c)}, missing {len(missing)}")
        cols = [x for x in ["question_id", "question_type", "corrective_action", "correction_reason", "error"] if x in r.columns]
        say(missing[cols].to_string()[:3000])
        say(f"missing ids with empty answer: {int(missing['answer'].str.strip().eq('').sum())}; "
            f"empty draft: {int(missing['draft_answer'].str.strip().eq('').sum()) if 'draft_answer' in missing else 'n/a'}")
    show_lines("analyse_latency.py", 1, 45)


@guarded
def item11_check_project():
    section("ITEM 11. FILES READ BY check_project.py")
    grep("check_project.py", r"\.csv|RESULTS|results")


@guarded
def item12_validation_sample():
    section("ITEM 12. VALIDATION SAMPLE SOURCE")
    show_lines("src/build_validation_sample.py", 1, 59)


@guarded
def item13_corpus():
    section("ITEM 13/14. CORPUS INVENTORY, EXTRACTION METHOD, QUESTION AUDIT")
    for name in ["data/document_inventory.csv", "data/document_inventory1.csv"]:
        p = ROOT / name
        if p.exists():
            d = read_csv(p)
            say(f"\n{name}: rows {len(d)}; columns {list(d.columns)}")
            say(d.head(20).to_string()[:3500])
    grep("src/document_loader.py", r"import |pdfplumber|fitz|PyPDF|pypdf|ocr|OCR|def |exclude|skip", 60)
    a, b = ROOT / "data/evaluation/test_questions_v2_answerable_pre_audit.csv", ROOT / "data/evaluation/test_questions_v2_answerable.csv"
    if a.exists() and b.exists():
        pa, pb = read_csv(a).set_index("question_id"), read_csv(b).set_index("question_id")
        common = pa.index.intersection(pb.index)
        say(f"\nAudit diff: {len(common)} shared ids; only pre-audit {len(pa.index.difference(pb.index))}; "
            f"only post-audit {len(pb.index.difference(pa.index))}")
        for col in pa.columns.intersection(pb.columns):
            changed = (pa.loc[common, col] != pb.loc[common, col]).sum()
            if changed:
                say(f"  {col}: {int(changed)} rows changed")


@guarded
def item16_hygiene():
    section("ITEM 16. .gitignore, tests/, README")
    t = text(".gitignore")
    say(".gitignore:\n" + (t[:1500] if t else "NOT FOUND"))
    env_ignored = bool(t and re.search(r"^\s*/?\.env\s*$", t, flags=re.M))
    say(f".env listed in .gitignore: {env_ignored}")
    tests = ROOT / "tests"
    if tests.exists():
        say("tests/: " + ", ".join(f"{p.name} ({p.stat().st_size} B)" for p in tests.rglob("*") if p.is_file()))
    try:
        r = subprocess.run(["git", "ls-files", ".env", "data/raw_documents"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        tracked = [l for l in r.stdout.splitlines() if l.strip()]
        say(f"git tracks .env: {'.env' in tracked}; tracked raw PDFs: {len([l for l in tracked if l.endswith('.pdf')])}")
        r = subprocess.run(["git", "remote", "-v"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        say("git remotes: " + (r.stdout.strip().replace("\n", " | ") or "(none)"))
    except Exception as e:
        say(f"git not available: {e}")
    t = text("README.md")
    if t:
        say("\nREADME.md (first 60 lines):\n" + "\n".join(t.splitlines()[:60]))


if __name__ == "__main__":
    steps = [item2_model, item3_refusal, item4_retrieval, item5_prompts, item6_failure_rules,
             item7_retriever_details, item8_dilution, item9_sweep, item10_missing,
             item11_check_project, item12_validation_sample, item13_corpus, item16_hygiene]
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step.__name__}...", flush=True)
        step()
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nDone. Upload {OUT.name} ({OUT.stat().st_size // 1024} KB) to Claude.")