"""
RAG Assistant with Corrective Hallucination Detection.

Pipeline:
    BM25 top-50 -> CrossEncoder rerank -> relevance threshold
    -> cited generation -> claim-level NLI verification
    -> corrective action (repair citations, or refuse)
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retriever import BM25Retriever  # noqa: E402

CANDIDATE_K = 50
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
RESULTS_DIR = PROJECT_ROOT / "results"
RETRIEVAL_RESULTS = RESULTS_DIR / "retrieval_evaluation_final_reranked.csv"

# Result files used in the dissertation (current verifier, rebuilt summaries).
S3_ANSWERS = RESULTS_DIR / "answers_s3_corrective_rag_clean.csv"
CLAIM_SUMMARY = RESULTS_DIR / "hallucination_evaluation_final_v2.csv"
S3_DRAFT_SUMMARY = RESULTS_DIR / "hallucination_evaluation_s3_drafts.csv"
EVALUATED_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

UNSUPPORTED_TRIGGER = 0.40

LABEL_STYLE = {
    "supported":         ("\U0001F7E2", "Supported"),
    "supported_uncited": ("\U0001F7E1", "Supported but uncited"),
    "contradicted":      ("\U0001F534", "Contradicted"),
    "unsupported":       ("\U0001F7E0", "Unsupported"),
    "nei":               ("\u26AA", "No verifiable claim"),
}

st.set_page_config(page_title="RAG Assistant - UoL Policies", layout="wide")


def fmt(value, digits=2, missing="n/a"):
    """Format a number that the verifier may leave undefined (None or NaN)."""
    if value is None:
        return missing
    try:
        if pd.isna(value):
            return missing
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return missing


def exceeds(value, threshold):
    """True only if value is a defined number above threshold."""
    try:
        return value is not None and not pd.isna(value) and float(value) > threshold
    except (TypeError, ValueError):
        return False


@st.cache_resource(show_spinner="Loading BM25 index...")
def get_retriever():
    return BM25Retriever()


@st.cache_resource(show_spinner="Loading CrossEncoder...")
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CE_MODEL)


@st.cache_data
def load_csv(path):
    p = Path(path)
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def get_llm_config():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        return None
    return {
        "provider": "NVIDIA",
        "key": key,
        "model": os.getenv("NVIDIA_MODEL", EVALUATED_MODEL),
        "base_url": NVIDIA_BASE_URL,
    }


def build_prompt(question, evidence):
    context = "\n\n".join(
        f"[{i}] ({row['document_name']}, page {row['page_number']})\n{row['chunk_text']}"
        for i, (_, row) in enumerate(evidence.iterrows(), start=1)
    )
    return (
        "You are a University of Liverpool student support assistant.\n\n"
        "Answer the question using ONLY the numbered evidence below.\n"
        "Rules:\n"
        "- Every sentence that states a fact must end with a citation, e.g. [2].\n"
        "- Cite only the specific evidence that supports that sentence.\n"
        "- Summary or concluding sentences also need citations - cite the evidence "
        "they summarise.\n"
        "- If the evidence does not contain the answer, say so explicitly.\n"
        "- Do not add information that is not in the evidence.\n"
        "- If the question contains a false assumption, correct it.\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


DEMO_MAX_TOKENS = 4096  # evaluation used 900; reasoning models need more room


def generate_answer(question, evidence, config):
    """Return (answer_text, diagnostics). The answer never includes model reasoning."""
    from openai import OpenAI

    client = OpenAI(api_key=config["key"], base_url=config["base_url"])
    messages = [{"role": "user", "content": build_prompt(question, evidence)}]
    if "gpt-oss" in config["model"]:
        # gpt-oss models read their reasoning effort from the system message.
        messages.insert(0, {"role": "system", "content": "Reasoning: low"})

    resp = client.chat.completions.create(
        model=config["model"],
        max_tokens=DEMO_MAX_TOKENS,
        temperature=0.0,
        messages=messages,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    reasoning = (
        getattr(choice.message, "reasoning_content", None)
        or getattr(choice.message, "reasoning", None)
        or ""
    )
    usage = getattr(resp, "usage", None)
    diagnostics = {
        "finish_reason": choice.finish_reason,
        "reasoning_chars": len(reasoning),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    return text, diagnostics


def repair_citations(answer, verification, evidence):
    """
    Insert citations for claims the verifier found grounded but uncited.

    Returns (repaired_answer, n_repaired).
    """
    chunk_to_position = {
        str(row["chunk_id"]): i
        for i, (_, row) in enumerate(evidence.iterrows(), start=1)
    }

    repaired = answer
    n = 0

    for claim in verification["claims"]:
        if claim.get("label") != "supported_uncited":
            continue

        position = chunk_to_position.get(str(claim.get("supporting_chunk")))
        if position is None:
            continue

        sentence = claim.get("sentence") or ""
        if not sentence or sentence not in repaired:
            continue

        if sentence.rstrip().endswith((".", "!", "?")):
            fixed = sentence.rstrip()[:-1].rstrip() + f" [{position}]."
        else:
            fixed = sentence.rstrip() + f" [{position}]"

        # Skip if the same marker already follows the sentence (e.g. "... advice. [1]"),
        # which happens when a model places citations after the full stop.
        start = repaired.find(sentence)
        following = repaired[start + len(sentence): start + len(sentence) + 12]
        if f"[{position}]" in following:
            continue

        repaired = repaired.replace(sentence, fixed, 1)
        n += 1

    return repaired, n


# --------------------------------------------------------------- sidebar
st.sidebar.title("Pipeline")
st.sidebar.markdown(
    f"""
**1. Retrieval** - BM25 sparse lexical, top-{CANDIDATE_K}  
**2. Reranking** - CrossEncoder cross-attention  
**3. Filtering** - drop chunks below a relevance threshold  
**4. Generation** - cited answer over surviving evidence  
**5. Verification** - claim-level NLI entailment  
**6. Correction** - repair citations, or refuse

Corpus: 1,395 chunks across 12 policy documents
"""
)
top_k = st.sidebar.slider("Maximum evidence chunks (k)", 3, 10, 5)
ce_threshold = st.sidebar.slider(
    "CrossEncoder relevance threshold", -5.0, 2.0, -1.0, 0.5
)
st.sidebar.caption(
    "Chunks below the threshold are withheld rather than padding to a fixed k."
)

config = get_llm_config()
if config:
    st.sidebar.success(f"LLM: {config['provider']}")
    st.sidebar.caption(config["model"])
    if config["model"] != EVALUATED_MODEL:
        st.sidebar.caption(
            f"Evaluated results in the other tabs were produced with "
            f"{EVALUATED_MODEL}."
        )
else:
    st.sidebar.warning("No NVIDIA_API_KEY found - retrieval only")

# --------------------------------------------------------------- main
st.title("RAG Assistant with Corrective Hallucination Detection")
st.caption("Every pipeline stage is logged so failures can be attributed to a source.")
st.info(
    "**Research prototype - not official University advice.** Answers and claim "
    "verdicts can be wrong. Always check the linked policy documents or contact "
    "Student Services before acting on an answer."
)

tab_demo, tab_systems, tab_eval, tab_diag = st.tabs(
    ["Ask a question", "System comparison", "Retrieval results", "Failure diagnosis"]
)

with tab_demo:
    examples = [
        "",
        "How should students or staff raise concerns about harassment, bullying, "
        "discrimination or victimisation?",
        "What support is available for international students on a Student Route visa?",
        "What participation is expected from students involved in a student-conduct "
        "investigation?",
        "What happens if I do not pay my tuition fees on time?",
        "Since attendance is not monitored at all, can I skip every lecture?",
        "How do the Undergraduate and Postgraduate Handbooks differ on extenuating "
        "circumstances?",
    ]
    picked = st.selectbox("Example questions", examples)
    question = st.text_input("Your question", value=picked)

    if st.button("Run pipeline", type="primary") and question.strip():
        retriever = get_retriever()
        reranker = get_reranker()

        t0 = time.perf_counter()
        pool = retriever.retrieve(question, top_k=CANDIDATE_K).copy()
        t_retrieval = time.perf_counter() - t0

        t1 = time.perf_counter()
        pool["ce_score"] = reranker.predict(
            [[question, str(t)] for t in pool["chunk_text"]]
        )
        reranked = pool.sort_values("ce_score", ascending=False).reset_index(drop=True)
        t_rerank = time.perf_counter() - t1

        candidates = reranked.head(top_k)
        evidence = candidates[candidates["ce_score"] > ce_threshold]
        below_threshold = evidence.empty
        if below_threshold:
            evidence = reranked.head(1)

        answer, gen_diag = "", {}
        if config:
            with st.spinner("Generating answer..."):
                try:
                    answer, gen_diag = generate_answer(question, evidence, config)
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        st.session_state["run"] = {
            "question": question,
            "pool": pool,
            "reranked": reranked,
            "candidates": candidates,
            "evidence": evidence,
            "withheld": len(candidates) - len(evidence),
            "below_threshold": below_threshold,
            "answer": answer,
            "gen_diag": gen_diag,
            "t_retrieval": t_retrieval,
            "t_rerank": t_rerank,
        }
        st.session_state.pop("verification", None)

    run = st.session_state.get("run")

    if run:
        evidence = run["evidence"]
        pool = run["pool"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Candidates", len(pool))
        c2.metric("Documents routed", pool["document_name"].nunique())
        c3.metric("Evidence used", f"{len(evidence)}/{top_k}")
        c4.metric("Retrieval", f"{run['t_retrieval']:.2f}s")
        c5.metric("Reranking", f"{run['t_rerank']:.2f}s")

        if run["below_threshold"]:
            st.warning(
                "No chunk passed the relevance threshold - treat this answer as "
                "low confidence."
            )
        elif run["withheld"]:
            st.caption(
                f"{run['withheld']} chunk(s) scored below the relevance threshold "
                f"({ce_threshold}) and were withheld from the model."
            )

        if config and not run["answer"].strip():
            diag = run.get("gen_diag") or {}
            reason = diag.get("finish_reason")
            hint = (
                "The model used its whole token budget on reasoning before writing an "
                "answer. Run the pipeline again, or use a non-reasoning model."
                if reason == "length" else
                "The model returned reasoning but no final answer. Run the pipeline "
                "again, or use a non-reasoning model."
                if diag.get("reasoning_chars") else
                "Check the model name in .env and try again."
            )
            st.warning(
                "The model returned an empty answer, so there are no claims to verify. "
                "The corrective layer treats this as unanswered.\n\n"
                f"**Diagnosis:** finish reason `{reason}`, "
                f"{diag.get('completion_tokens', '?')} completion tokens, "
                f"{diag.get('reasoning_chars', 0)} characters of reasoning. {hint}"
            )

        if run["answer"].strip():
            st.subheader("Draft answer")
            st.markdown(run["answer"])
            st.caption("Bracketed numbers refer to the evidence chunks below.")

            st.divider()
            st.subheader("Claim verification (S3 - Corrective layer)")

            if "verification" not in st.session_state:
                st.info(
                    "Each sentence is checked against the retrieved evidence using an "
                    "NLI entailment model. On CPU this takes roughly a minute."
                )
                if st.button("Verify claims", type="secondary"):
                    from src.faithfulness import verify_answer

                    evidence_list = [
                        {"chunk_id": str(r["chunk_id"]),
                         "chunk_text": str(r["chunk_text"])}
                        for _, r in evidence.iterrows()
                    ]
                    with st.spinner("Verifying claims against evidence..."):
                        st.session_state["verification"] = verify_answer(
                            run["answer"], evidence_list
                        )
                    st.rerun()
            else:
                v = st.session_state["verification"]
                n_contra = v.get("n_contradicted") or 0
                n_unsup = v.get("n_unsupported") or 0
                n_uncited = v.get("n_supported_uncited") or 0
                rate = v.get("unsupported_claim_rate")
                risk = v.get("hallucination_risk") or "N/A"

                risk_colour = {
                    "LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error
                }.get(risk, st.info)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Faithfulness", fmt(v.get("faithfulness")))
                m2.metric("Unsupported rate", fmt(rate))
                m3.metric("Citation accuracy", fmt(v.get("citation_accuracy")))
                m4.metric("Claims checked", v.get("n_checkable") or 0)

                if not v.get("n_checkable"):
                    st.caption(
                        "No checkable claims were found, so the rates above are "
                        "undefined."
                    )
                elif v.get("citation_accuracy") is None:
                    st.caption(
                        "Citation accuracy is undefined: no supported claim carried a "
                        "citation."
                    )

                should_refuse = (
                    n_contra > 0
                    or exceeds(rate, UNSUPPORTED_TRIGGER)
                    or run["below_threshold"]
                )

                risk_colour(
                    f"Hallucination risk: **{risk}** - "
                    + ("the corrective layer would **refuse** this answer."
                       if should_refuse else
                       "the corrective layer would **accept** this answer.")
                )
                st.caption(
                    "Claim verdicts come from an automatic NLI model and are often "
                    "wrong; use them as prompts to check the evidence, not as "
                    "judgements."
                )

                # ---- per-question claim-origin attribution (heuristic) ----
                ce_scores = evidence["ce_score"].tolist()
                best_ce = max(ce_scores) if ce_scores else -99.0
                ungrounded = n_unsup + n_contra

                if ungrounded == 0:
                    stage, detail = "None", (
                        "No checkable claim was flagged as ungrounded."
                    )
                elif run["below_threshold"]:
                    stage, detail = "Retrieval", (
                        "No passage passed the relevance threshold, so the model had "
                        "little relevant evidence to ground an answer on."
                    )
                elif best_ce < 0:
                    stage, detail = "Retrieval", (
                        f"The best retrieved passage scored {best_ce:.2f}, so the "
                        "reranker judged nothing retrieved to be strongly relevant."
                    )
                elif n_contra > 0:
                    stage, detail = "Generation", (
                        f"Relevant-looking evidence was retrieved (best CE score "
                        f"{best_ce:.2f}), but {n_contra} claim(s) were labelled "
                        "contradicted."
                    )
                else:
                    stage, detail = "Generation", (
                        f"Relevant-looking evidence was retrieved (best CE score "
                        f"{best_ce:.2f}), but {n_unsup} claim(s) were labelled "
                        "unsupported."
                    )

                st.markdown("**Where might the flagged claims originate? (heuristic)**")
                f1, f2 = st.columns([1, 3])
                f1.metric("Likely stage", stage)
                f2.info(f"**{stage}** - {detail}")
                st.caption(
                    "Heuristic based on reranker scores. In evaluation it agreed "
                    "poorly with gold-label attribution, because relevance scores "
                    "do not show whether the evidence is sufficient. Treat it as "
                    "indicative only."
                )

                st.markdown("**Per-claim verdict**")
                for claim in v.get("claims", []):
                    icon, label = LABEL_STYLE.get(
                        claim.get("label"), ("\u2753", str(claim.get("label")))
                    )
                    cited_list = claim.get("cited") or []
                    cited = (
                        ", ".join(f"[{c}]" for c in cited_list)
                        if cited_list else "no citation"
                    )
                    sentence = claim.get("sentence") or ""
                    with st.expander(f"{icon} {label} - {sentence[:90]}..."):
                        st.write(sentence)
                        st.caption(
                            f"Cited: {cited} | entailment "
                            f"{fmt(claim.get('best_entailment'), 3)} | contradiction "
                            f"{fmt(claim.get('best_contradiction'), 3)} | best "
                            f"supporting chunk: {claim.get('supporting_chunk') or '-'}"
                        )

                if n_uncited > 0:
                    repaired, n_fixed = repair_citations(run["answer"], v, evidence)
                    if n_fixed:
                        st.divider()
                        st.subheader("Corrective action 1 - citation repair")
                        st.success(
                            f"{n_fixed} claim(s) judged grounded were missing a "
                            "citation. The corrective layer inserted the chunk the "
                            "verifier found to entail each claim:"
                        )
                        st.markdown(repaired)
                        st.caption(
                            "Citation repair is shown in this interface only; it was "
                            "not part of the evaluated system."
                        )

                if should_refuse:
                    st.divider()
                    st.subheader("Corrective action 2 - refusal")
                    st.error(
                        "I cannot answer this question reliably from the available "
                        "University documents. The retrieved evidence does not "
                        "adequately support a complete answer. Please consult the "
                        "relevant policy document directly, or contact Student "
                        "Services."
                    )

        st.divider()
        st.subheader(f"Evidence supplied to the model ({len(evidence)})")
        for i, (_, row) in enumerate(evidence.iterrows(), start=1):
            with st.expander(
                f"[{i}] {row['document_name']} - page {row['page_number']} "
                f"| CE score {row['ce_score']:.3f}"
            ):
                st.write(row["chunk_text"])
                url = str(row.get("source_url", "")).strip()
                if url.startswith("http"):
                    st.markdown(
                        f"[Open source document]({url}) - page {row['page_number']}"
                    )
                st.caption(f"chunk_id: {row['chunk_id']}")

        if run["withheld"]:
            with st.expander(f"Withheld below threshold ({run['withheld']})"):
                for _, row in run["candidates"][
                    run["candidates"]["ce_score"] <= ce_threshold
                ].iterrows():
                    st.markdown(
                        f"**{row['document_name']}** - page {row['page_number']} "
                        f"| CE score {row['ce_score']:.3f}"
                    )
                    st.caption(str(row["chunk_text"])[:400] + "...")

        with st.expander("Retrieval trace - BM25 rank vs reranked position"):
            trace = run["reranked"].head(20)[
                ["chunk_id", "document_name", "page_number", "ce_score"]
            ].copy()
            trace.insert(0, "reranked_position", range(1, len(trace) + 1))
            bm25_order = pool["chunk_id"].astype(str).tolist()
            trace["bm25_rank"] = [
                bm25_order.index(str(c)) + 1 for c in trace["chunk_id"]
            ]
            trace["movement"] = trace["bm25_rank"] - trace["reranked_position"]
            st.dataframe(trace, width="stretch")
            st.caption("Positive movement = promoted by the CrossEncoder.")

with tab_systems:
    s3 = load_csv(S3_ANSWERS)
    hall = load_csv(CLAIM_SUMMARY)
    drafts = load_csv(S3_DRAFT_SUMMARY)

    st.subheader("S3 - effect of the corrective layer")
    st.caption(
        f"Evaluated with {EVALUATED_MODEL}. Questions with an empty draft are excluded."
    )

    if s3 is None:
        st.info(f"{S3_ANSWERS.name} not found.")
    else:
        action = s3["corrective_action"].astype(str).str.lower()
        refused = s3[action == "refused"]
        accepted = s3[action == "accepted"]

        delivered_rate = None
        if hall is not None:
            s3_rows = hall[hall["system"].astype(str).str.startswith("S3")]
            delivered_rate = s3_rows["unsupported_claim_rate"].mean()
        withheld_rate = (
            drafts["unsupported_claim_rate"].mean() if drafts is not None else None
        )

        a, b, c, d = st.columns(4)
        a.metric("Answered questions", len(s3))
        b.metric("Withheld", len(refused))
        c.metric("Ungrounded rate, withheld drafts", fmt(withheld_rate, 3, "-"))
        d.metric("Ungrounded rate, delivered answers", fmt(delivered_rate, 3, "-"))

        st.caption(
            f"{len(refused)} of {len(s3)} drafts were withheld. Delivered answers score "
            "as better grounded largely because the gate selects them on this measure; "
            "the trade-off is reduced coverage."
        )

        st.markdown("**Corrective action by question type**")
        by_type = pd.crosstab(s3["question_type"], action)
        st.dataframe(by_type, width="stretch")

        afp = s3[s3["question_type"] == "adversarial_false_premise"]
        if len(afp):
            n_afp_refused = int((afp["corrective_action"].str.lower() == "refused").sum())
            with st.expander("Known limitation - false-premise questions"):
                st.markdown(
                    f"{n_afp_refused} of {len(afp)} adversarial false-premise questions "
                    "were withheld. A likely cause is that a correct answer rejecting a "
                    "false premise asserts that something is *not* in the policy, which "
                    "no passage can entail, and its framing sentences are often "
                    "labelled contradicted. Rewriting claims to stand alone and "
                    "verifying them against combined evidence may help."
                )

        st.markdown("**Withheld examples - draft and trigger**")
        for _, row in refused.head(3).iterrows():
            with st.expander(f"{row['question_id']} - {str(row['question'])[:80]}..."):
                st.markdown("**Draft (would have been shown):**")
                st.write(row["draft_answer"])
                st.markdown("**Trigger:**")
                st.warning(row["correction_reason"])

    if hall is not None:
        st.divider()
        st.subheader("Claim-level results by system")
        summary = hall.groupby("system").agg(
            answers_scored=("faithfulness", "count"),
            claims=("n_checkable", "sum"),
            faithfulness=("faithfulness", "mean"),
            ungrounded_rate=("unsupported_claim_rate", "mean"),
            citation_accuracy=("citation_accuracy", "mean"),
        ).round(3)
        st.dataframe(summary, width="stretch")
        st.caption(
            "All 102 answerable questions. S0 has no retrieval, so its claims are "
            "checked against the gold passages. S3 rows cover delivered answers only. "
            "All values are the verifier's judgement, which a blind human check found "
            "unreliable at claim level."
        )

with tab_eval:
    df = load_csv(RETRIEVAL_RESULTS)
    if df is None:
        st.info("Run `python -m src.evaluate_final_retriever` first.")
    else:
        st.subheader("Frozen retriever - 102 answerable questions")
        a, b, c, d, e, f = st.columns(6)
        a.metric("Hit@5", f"{df['hit_at_5'].mean():.3f}")
        b.metric("Recall@5", f"{df['recall_at_5'].mean():.3f}")
        c.metric("Recall@10", f"{df['recall_at_10'].mean():.3f}")
        d.metric("Precision@5", f"{df['precision_at_5'].mean():.3f}")
        e.metric("nDCG@5", f"{df['ndcg_at_5'].mean():.3f}")
        f.metric("MRR (full list)", f"{df['mrr'].mean():.3f}")
        st.subheader("BM25 baseline vs CrossEncoder rerank")
        by_type = df.groupby("question_type").agg(
            n=("question_id", "count"),
            bm25_hit5=("bm25_hit_at_5", "mean"),
            reranked_hit5=("hit_at_5", "mean"),
            reranked_hit10=("hit_at_10", "mean"),
        ).round(3)
        by_type["improvement"] = (
            by_type["reranked_hit5"] - by_type["bm25_hit5"]
        ).round(3)
        st.dataframe(by_type, width="stretch")

        worse = by_type[by_type["improvement"] < 0]
        best_type = by_type["improvement"].idxmax()
        st.caption(
            (f"Reranking lowered Hit@5 for: {', '.join(worse.index)}. "
             if len(worse) else "Reranking improved or held Hit@5 for every type. ")
            + f"Largest gain: {best_type} "
            f"(+{by_type.loc[best_type, 'improvement']:.3f})."
        )

        st.subheader("Recall at increasing k")
        recall_by_k = pd.DataFrame(
            {"k": [1, 3, 5, 10],
             "recall": [df[f"recall_at_{k}"].mean() for k in [1, 3, 5, 10]]}
        )
        st.bar_chart(recall_by_k, x="k", y="recall")
        st.dataframe(recall_by_k.round(3).set_index("k").T, width="stretch")

with tab_diag:
    df = load_csv(RETRIEVAL_RESULTS)
    if df is None:
        st.info("Run `python -m src.evaluate_final_retriever` first.")
    else:
        st.subheader("Where does each retrieval failure originate?")
        st.markdown(
            "Gold evidence is traced through the pipeline: if it never enters the "
            "candidate pool the failure is **retrieval**; if it enters but misses the "
            "final cut the failure is **ranking**."
        )

        def classify(row):
            if row["recall_at_5"] == 1.0:
                return "Retrieval success"
            if row["gold_in_pool"] == 0.0:
                return "Retrieval failure"
            if row["recall_at_5"] > 0:
                return "Partial - ranking failure"
            return "Ranking failure"

        df = df.copy()
        df["diagnosis"] = df.apply(classify, axis=1)

        counts = df["diagnosis"].value_counts()
        cols = st.columns(len(counts))
        for col, (label, n) in zip(cols, counts.items()):
            col.metric(label, f"{n}/{len(df)}")
        st.bar_chart(counts)

        options = sorted(df["diagnosis"].unique())
        choice = st.multiselect("Filter", options, default=options)
        st.dataframe(
            df[df["diagnosis"].isin(choice)][
                ["question_id", "question_type", "difficulty", "diagnosis",
                 "gold_in_pool", "recall_at_5", "recall_at_10"]
            ],
            width="stretch",
        )

        st.info(
            "**Scope.** This diagnosis is retrospective and uses gold labels. The "
            "stage estimate on the first tab needs no gold labels, but it is a "
            "heuristic that agreed poorly with gold-label attribution in evaluation."
        )