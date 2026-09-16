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

UNSUPPORTED_TRIGGER = 0.40

LABEL_STYLE = {
    "supported":         ("\U0001F7E2", "Supported"),
    "supported_uncited": ("\U0001F7E1", "Supported but uncited"),
    "contradicted":      ("\U0001F534", "Contradicted"),
    "unsupported":       ("\U0001F7E0", "Unsupported"),
    "nei":               ("\u26AA", "No verifiable claim"),
}

st.set_page_config(page_title="RAG Assistant - UoL Policies", layout="wide")


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
        "model": os.getenv("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2"),
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


def generate_answer(question, evidence, config):
    from openai import OpenAI

    client = OpenAI(api_key=config["key"], base_url=config["base_url"])
    resp = client.chat.completions.create(
        model=config["model"],
        max_tokens=900,
        temperature=0.0,
        messages=[{"role": "user", "content": build_prompt(question, evidence)}],
    )
    return resp.choices[0].message.content


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
        if claim["label"] != "supported_uncited":
            continue

        position = chunk_to_position.get(str(claim["supporting_chunk"]))
        if position is None:
            continue

        sentence = claim["sentence"]
        if not sentence or sentence not in repaired:
            continue

        if sentence.rstrip().endswith((".", "!", "?")):
            fixed = sentence.rstrip()[:-1].rstrip() + f" [{position}]."
        else:
            fixed = sentence.rstrip() + f" [{position}]"

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
else:
    st.sidebar.warning("No NVIDIA_API_KEY found - retrieval only")

# --------------------------------------------------------------- main
st.title("RAG Assistant with Corrective Hallucination Detection")
st.caption("Every pipeline stage is logged so failures can be attributed to a source.")

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

        answer = ""
        if config:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(question, evidence, config)
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

        if run["answer"]:
            st.subheader("Draft answer (S2 - Modular RAG)")
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

                risk_colour = {
                    "LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error
                }[v["hallucination_risk"]]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Faithfulness", f"{v['faithfulness']:.2f}")
                m2.metric("Unsupported rate", f"{v['unsupported_claim_rate']:.2f}")
                m3.metric("Citation accuracy", f"{v['citation_accuracy']:.2f}")
                m4.metric("Claims checked", v["n_checkable"])

                should_refuse = (
                    v["n_contradicted"] > 0
                    or v["unsupported_claim_rate"] > UNSUPPORTED_TRIGGER
                    or run["below_threshold"]
                )

                risk_colour(
                    f"Hallucination risk: **{v['hallucination_risk']}** - "
                    + ("the corrective layer would **refuse** this answer."
                       if should_refuse else
                       "the corrective layer would **accept** this answer.")
                )

                # ---- per-question claim-origin attribution ----
                ce_scores = evidence["ce_score"].tolist()
                best_ce = max(ce_scores) if ce_scores else -99.0
                ungrounded = v["n_unsupported"] + v["n_contradicted"]

                if ungrounded == 0:
                    stage, detail = "None", (
                        "Every checkable claim is entailed by the retrieved evidence."
                    )
                elif run["below_threshold"]:
                    stage, detail = "Retrieval", (
                        "No passage passed the relevance threshold, so the model had "
                        "no adequate evidence to ground an answer on - the ungrounded "
                        "claims originate at the retrieval stage."
                    )
                elif best_ce < 0:
                    stage, detail = "Retrieval", (
                        f"The best retrieved passage scored {best_ce:.2f} - the "
                        "reranker judged nothing in the corpus to be strongly relevant "
                        "to this question, so the ungrounded claims originate at the "
                        "retrieval stage."
                    )
                elif v["n_contradicted"] > 0:
                    stage, detail = "Generation", (
                        f"Relevant evidence was retrieved (best CE score "
                        f"{best_ce:.2f}), but {v['n_contradicted']} claim(s) "
                        "contradict it - these originate at the generation stage, "
                        "not retrieval."
                    )
                else:
                    stage, detail = "Generation", (
                        f"Relevant evidence was retrieved (best CE score "
                        f"{best_ce:.2f}), but {v['n_unsupported']} claim(s) go beyond "
                        "what it states - these originate at the generation stage, "
                        "not retrieval."
                    )

                st.markdown("**Where do the ungrounded claims originate?**")
                f1, f2 = st.columns([1, 3])
                f1.metric("Stage", stage)
                f2.info(f"**{stage}** - {detail}")
                st.caption(
                    "Attributed at runtime from reranker scores and claim "
                    "verification - no gold labels required."
                )

                st.markdown("**Per-claim verdict**")
                for claim in v["claims"]:
                    icon, label = LABEL_STYLE[claim["label"]]
                    cited = (
                        ", ".join(f"[{c}]" for c in claim["cited"])
                        if claim["cited"] else "no citation"
                    )
                    with st.expander(
                        f"{icon} {label} - {claim['sentence'][:90]}..."
                    ):
                        st.write(claim["sentence"])
                        st.caption(
                            f"Cited: {cited} | entailment "
                            f"{claim['best_entailment']:.3f} | contradiction "
                            f"{claim['best_contradiction']:.3f} | best supporting "
                            f"chunk: {claim['supporting_chunk']}"
                        )

                if v["n_supported_uncited"] > 0:
                    repaired, n_fixed = repair_citations(run["answer"], v, evidence)
                    if n_fixed:
                        st.divider()
                        st.subheader("Corrective action 1 - citation repair")
                        st.success(
                            f"{n_fixed} grounded claim(s) were missing a citation. "
                            "The corrective layer inserted the chunk the verifier "
                            "found to entail each claim:"
                        )
                        st.markdown(repaired)
                        st.caption(
                            "Repair is a second corrective action alongside refusal: "
                            "claims that are true but unsourced are fixed rather than "
                            "discarded."
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
    s3 = load_csv(RESULTS_DIR / "answers_s3_corrective_rag.csv")
    hall = load_csv(RESULTS_DIR / "hallucination_evaluation.csv")

    st.subheader("S3 - effect of the corrective layer")
    st.caption("Full evaluation - 102 answerable questions.")

    if s3 is None:
        st.info("Run `python -m src.evaluate_s3_corrective_rag` first.")
    else:
        refused = s3[s3["corrective_action"] == "refused"]
        accepted = s3[s3["corrective_action"] == "accepted"]
        prevented = int(
            refused["draft_n_unsupported"].sum() + refused["draft_n_contradicted"].sum()
        )

        a, b, c, d = st.columns(4)
        a.metric("Questions", len(s3))
        b.metric("Draft unsupported rate", f"{s3['draft_unsupported_rate'].mean():.3f}")
        c.metric(
            "After correction",
            f"{accepted['draft_unsupported_rate'].mean():.3f}" if len(accepted) else "-",
        )
        d.metric("Claims withheld", prevented)

        st.caption(
            f"{len(refused)} of {len(s3)} answers were refused. Unsupported claim rate "
            "among delivered answers falls accordingly - the trade-off is reduced "
            "coverage."
        )

        st.markdown("**Draft risk vs corrective action**")
        st.dataframe(
            pd.crosstab(s3["draft_risk"], s3["corrective_action"]), width="stretch"
        )

        st.markdown("**Corrective action by question type**")
        st.dataframe(
            pd.crosstab(s3["question_type"], s3["corrective_action"]), width="stretch"
        )

        with st.expander("Known limitation - negation and reported speech"):
            st.markdown(
                "All four adversarial false-premise questions were refused. The "
                "drafts correctly *refuted* the false premise, but the NLI verifier "
                "cannot distinguish a claim that is asserted from one that is quoted "
                "in order to be rejected, so it scores the refutation as a "
                "contradiction. Decomposing sentences into atomic claims before "
                "verification (as in FActScore) would address this."
            )

        st.markdown("**Caught examples - draft vs delivered**")
        for _, row in refused.head(3).iterrows():
            with st.expander(f"{row['question_id']} - {row['question'][:80]}..."):
                st.markdown("**Draft (would have been shown):**")
                st.write(row["draft_answer"])
                st.markdown("**Trigger:**")
                st.warning(row["correction_reason"])

    if hall is not None:
        st.divider()
        st.subheader("Claim-level faithfulness by system")

        # S0 has no citations and no retrieved evidence, so its verification
        # requires comparison against gold chunks - that run is still pending.
        hall_valid = hall[hall["system"] != "S0_llm_only"]

        st.dataframe(
            hall_valid.groupby("system").agg(
                questions=("question_id", "count"),
                claims=("n_checkable", "sum"),
                faithfulness=("faithfulness", "mean"),
                unsupported_rate=("unsupported_claim_rate", "mean"),
                citation_accuracy=("citation_accuracy", "mean"),
            ).round(3),
            width="stretch",
        )
        st.caption(
            "S0 (LLM-only) verification is pending: with no retrieved evidence, its "
            "claims must be checked against gold chunks rather than supplied context."
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
        f.metric("MRR", f"{df['mrr'].mean():.3f}")
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
        st.caption(
            "Reranking improves or holds every question type. The largest gain is on "
            "adversarial false-premise questions."
        )

        st.subheader("Recall at increasing k")
        st.bar_chart(
            pd.DataFrame(
                {"recall": [df[f"recall_at_{k}"].mean() for k in [1, 3, 5, 10]]},
                index=["k=1", "k=3", "k=5", "k=10"],
            )
        )

with tab_diag:
    df = load_csv(RETRIEVAL_RESULTS)
    if df is None:
        st.info("Run `python -m src.evaluate_final_retriever` first.")
    else:
        st.subheader("Where does each failure originate?")
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
            "**Scope.** Retrieval diagnosis is retrospective and uses gold labels. "
            "Per-question attribution on the demo tab runs without gold labels and "
            "works on unseen questions."
        )