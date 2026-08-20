"""
RAG Assistant with Retrieval Tracing — University of Liverpool policy corpus.

Frozen pipeline:
    BM25 top-50 -> CrossEncoder rerank -> relevance threshold -> cited answer
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
RESULTS_PATH = PROJECT_ROOT / "results" / "retrieval_evaluation_final_reranked.csv"

st.set_page_config(page_title="RAG Assistant — UoL Policies", layout="wide")


@st.cache_resource(show_spinner="Loading BM25 index...")
def get_retriever():
    return BM25Retriever()


@st.cache_resource(show_spinner="Loading CrossEncoder...")
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CE_MODEL)


@st.cache_data
def get_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
    return None


def get_llm_config():
    """NVIDIA NIM only. Ignores any OPENAI_API_KEY in the environment."""
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
        "- Cite the evidence number in square brackets after each claim, e.g. [2].\n"
        "- Cite only the specific evidence that supports that claim. Do not cite "
        "every item after every sentence.\n"
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


# --------------------------------------------------------------- sidebar
st.sidebar.title("Pipeline")
st.sidebar.markdown(
    f"""
**1. Retrieval** — BM25 sparse lexical, top-{CANDIDATE_K}  
**2. Reranking** — CrossEncoder cross-attention  
**3. Filtering** — drop chunks below a relevance threshold  
**4. Generation** — cited answer over surviving evidence

Corpus: 1,395 chunks across 12 policy documents
"""
)
top_k = st.sidebar.slider("Maximum evidence chunks (k)", 3, 10, 5)
ce_threshold = st.sidebar.slider(
    "CrossEncoder relevance threshold", -5.0, 2.0, -1.0, 0.5
)
st.sidebar.caption(
    "Chunks scoring below the threshold are withheld from the LLM rather than "
    "padding the context to a fixed k."
)

config = get_llm_config()
if config:
    st.sidebar.success(f"LLM: {config['provider']}")
    st.sidebar.caption(config["model"])
else:
    st.sidebar.warning("No NVIDIA_API_KEY found — retrieval only")

# --------------------------------------------------------------- main
st.title("RAG Assistant with Retrieval Tracing")
st.caption("Every pipeline stage is logged so failures can be attributed to a source.")

tab_demo, tab_eval, tab_diag = st.tabs(
    ["Ask a question", "Evaluation results", "Failure diagnosis"]
)

with tab_demo:
    examples = [
        "",
        "What happens if I do not pay my tuition fees on time?",
        "Since attendance is not monitored at all, can I skip every lecture?",
        "What are the sanctions for serious student misconduct?",
        "How do the Undergraduate and Postgraduate Handbooks differ on extenuating circumstances?",
        "What support is available for international students on a Student Route visa?",
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
        withheld = len(candidates) - len(evidence)

        if evidence.empty:
            evidence = reranked.head(1)
            st.warning(
                "No chunk passed the relevance threshold. Showing the single best "
                "candidate — treat this answer as low confidence."
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Candidates", len(pool))
        c2.metric("Documents routed", pool["document_name"].nunique())
        c3.metric("Evidence used", f"{len(evidence)}/{top_k}")
        c4.metric("Retrieval", f"{t_retrieval:.2f}s")
        c5.metric("Reranking", f"{t_rerank:.2f}s")

        if withheld:
            st.caption(
                f"{withheld} chunk(s) scored below the relevance threshold "
                f"({ce_threshold}) and were withheld from the model."
            )

        if config:
            st.subheader("Answer")
            with st.spinner("Generating..."):
                try:
                    st.markdown(generate_answer(question, evidence, config))
                    st.caption("Bracketed numbers refer to the evidence chunks below.")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        st.subheader(f"Evidence supplied to the model ({len(evidence)})")
        for i, (_, row) in enumerate(evidence.iterrows(), start=1):
            with st.expander(
                f"[{i}] {row['document_name']} — page {row['page_number']} "
                f"· CE score {row['ce_score']:.3f}"
            ):
                st.write(row["chunk_text"])
                st.caption(f"chunk_id: {row['chunk_id']}")

        if withheld:
            with st.expander(f"Withheld below threshold ({withheld})"):
                for _, row in candidates[
                    candidates["ce_score"] <= ce_threshold
                ].iterrows():
                    st.markdown(
                        f"**{row['document_name']}** — page {row['page_number']} "
                        f"· CE score {row['ce_score']:.3f}"
                    )
                    st.caption(str(row["chunk_text"])[:400] + "...")

        with st.expander("Retrieval trace — BM25 rank vs reranked position"):
            trace = reranked.head(20)[
                ["chunk_id", "document_name", "page_number", "ce_score"]
            ].copy()
            trace.insert(0, "reranked_position", range(1, len(trace) + 1))
            bm25_order = pool["chunk_id"].astype(str).tolist()
            trace["bm25_rank"] = [bm25_order.index(str(c)) + 1 for c in trace["chunk_id"]]
            trace["movement"] = trace["bm25_rank"] - trace["reranked_position"]
            st.dataframe(trace, width="stretch")
            st.caption(
                "Positive movement = promoted by the CrossEncoder. This trace is the "
                "basis for attributing a failure to retrieval or to ranking."
            )

with tab_eval:
    df = get_results()
    if df is None:
        st.info("Run `python -m src.evaluate_final_retriever` first.")
    else:
        st.subheader("Frozen pipeline — 102 answerable questions")
        a, b, c, d = st.columns(4)
        a.metric("Hit@5", f"{df['hit_at_5'].mean():.3f}")
        b.metric("Hit@10", f"{df['hit_at_10'].mean():.3f}")
        c.metric("Recall@5", f"{df['recall_at_5'].mean():.3f}")
        d.metric("MRR", f"{df['mrr'].mean():.3f}")

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
            "adversarial false-premise questions, where lexical matching retrieves the "
            "premise itself rather than the passage that refutes it."
        )

        st.subheader("Recall at increasing k")
        st.bar_chart(
            pd.DataFrame(
                {"recall": [df[f"recall_at_{k}"].mean() for k in [1, 3, 5, 10]]},
                index=["k=1", "k=3", "k=5", "k=10"],
            )
        )

with tab_diag:
    df = get_results()
    if df is None:
        st.info("Run `python -m src.evaluate_final_retriever` first.")
    else:
        st.subheader("Where does each failure originate?")
        st.markdown(
            "Each question is classified by tracing its gold evidence through the "
            "pipeline: if the evidence never enters the candidate pool the failure is "
            "**retrieval**; if it enters the pool but misses the final cut the failure "
            "is **ranking**."
        )

        def classify(row):
            if row["recall_at_5"] == 1.0:
                return "Retrieval success"
            if row["gold_in_pool"] == 0.0:
                return "Retrieval failure"
            if row["recall_at_5"] > 0:
                return "Partial — ranking failure"
            return "Ranking failure"

        df = df.copy()
        df["diagnosis"] = df.apply(classify, axis=1)

        counts = df["diagnosis"].value_counts()
        cols = st.columns(len(counts))
        for col, (label, n) in zip(cols, counts.items()):
            col.metric(label, f"{n}/{len(df)}")

        st.bar_chart(counts)

        st.subheader("Per-question diagnosis")
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
            "**Scope.** This diagnosis is retrospective: it uses gold chunk labels from "
            "the evaluation set. A runtime detector — classifying failures on unseen "
            "questions without gold labels, plus generation-faithfulness checking — is "
            "the next stage of the project."
        )