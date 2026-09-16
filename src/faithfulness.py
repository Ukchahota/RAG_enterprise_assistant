"""
Claim-level faithfulness verification.

Splits an answer into sentences, extracts [n] citations, and checks each
sentence against the supplied evidence using an NLI entailment model.

Labels (per proposal section 7.4):
    supported            - a cited chunk entails the sentence
    supported_uncited    - no/wrong citation, but some supplied chunk entails it
    contradicted         - a supplied chunk contradicts the sentence
    unsupported          - no supplied chunk entails the sentence
    nei                  - the sentence makes no verifiable factual claim

Abstentions are handled explicitly: the corrective layer's refusal template
describes the system's own verification state rather than asserting anything
about the policy corpus, so its sentences are excluded from n_checkable. An
abstaining answer therefore has no claim-level faithfulness score (the metric
is undefined, not zero) and is counted separately as coverage loss.
"""

import re
from functools import lru_cache

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

ENTAILMENT_THRESHOLD = 0.50
CONTRADICTION_THRESHOLD = 0.50

# Risk bands over the ungrounded (unsupported + contradicted) claim rate.
# NOTE: a single contradicted claim forces HIGH. Because the contradiction
# signal is low-precision on this corpus, HIGH counts are inflated by false
# contradictions; report the unsupported and contradicted components
# separately rather than relying on the risk band alone.
RISK_HIGH_RATE = 0.5
RISK_MEDIUM_RATE = 0.2

# Sentences that assert nothing checkable.
NON_CLAIM_PATTERNS = [
    r"^\s*(here|below|the following|in summary|to summari[sz]e)\b",
    r"^\s*(i hope|let me know|feel free|please note that if)\b",
    r"^\s*(the evidence (does not|doesn't) (contain|include|provide))",
    r"^\s*(based on the (provided )?evidence)[,:]?\s*$",
    r":\s*$",
]

# Abstention boilerplate emitted by the corrective layer (S3). Kept separate
# from NON_CLAIM_PATTERNS so an abstention can be distinguished from an empty
# or non-assertive answer: both yield zero checkable claims, but only the
# former is a deliberate refusal. Matched case-insensitively against the
# citation-stripped sentence.
ABSTENTION_PATTERNS = [
    r"^\s*i cannot answer this question reliably",
    r"^\s*the retrieved evidence does not adequately support a complete answer",
    r"^\s*please consult the relevant policy document",
]

_pipeline = None


def get_nli():
    """Load the NLI pipeline once per process."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=NLI_MODEL,
            top_k=None,
        )
    return _pipeline


def split_sentences(text: str) -> list[str]:
    """Conservative sentence splitter that protects [1][2] citation markers."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    protected = re.sub(
        r"\b(e\.g|i\.e|etc|vs|no|pp|approx)\.", r"\1<DOT>", text,
        flags=re.IGNORECASE,
    )
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    # Split on sentence ends and on bullet markers, so list stems don't fuse
    # with their first item.
    protected = re.sub(r"\s*[-*\u2022]\s+", " <BULLET> ", protected)
    parts = re.split(r"(?<=[.!?])\s+|<BULLET>", protected)
    out = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        part = re.sub(r"^[-*\u2022]\s*", "", part)
        part = re.sub(r"\*+", "", part)
        if len(part.split()) >= 4:
            out.append(part)
    return out


def extract_citations(sentence: str) -> list[int]:
    """Return the evidence numbers cited in a sentence, e.g. [1][3] -> [1, 3]."""
    found = []
    for group in re.findall(r"\[([\d,\s]+)\]", sentence):
        for token in group.split(","):
            token = token.strip()
            if token.isdigit():
                found.append(int(token))
    return sorted(set(found))


def strip_citations(sentence: str) -> str:
    """Remove citation markers so the NLI model sees clean text."""
    return re.sub(r"\s*\[[\d,\s]+\]", "", sentence).strip()


def is_abstention(sentence: str) -> bool:
    """True if the sentence is part of the corrective layer's refusal template."""
    low = str(sentence).lower()
    return any(re.search(p, low) for p in ABSTENTION_PATTERNS)


def is_non_claim(sentence: str) -> bool:
    """True if the sentence asserts nothing checkable against the corpus."""
    low = str(sentence).lower()
    if any(re.search(p, low) for p in NON_CLAIM_PATTERNS):
        return True
    return is_abstention(sentence)


@lru_cache(maxsize=20000)
def _nli_scores(premise: str, hypothesis: str) -> tuple[float, float, float]:
    """Return (entailment, neutral, contradiction) probabilities."""
    nli = get_nli()
    result = nli({"text": premise[:2000], "text_pair": hypothesis[:1000]})

    # Pipeline returns [{...}, ...] or [[{...}, ...]] depending on version.
    if result and isinstance(result[0], list):
        result = result[0]

    scores = {}
    for item in result:
        if isinstance(item, dict) and "label" in item:
            scores[item["label"].lower()] = item["score"]

    if not scores:
        raise ValueError(f"Unexpected NLI output shape: {result}")

    return (
        scores.get("entailment", 0.0),
        scores.get("neutral", 0.0),
        scores.get("contradiction", 0.0),
    )


def verify_sentence(sentence: str, evidence: list[dict]) -> dict:
    """
    Verify one sentence against a list of evidence dicts.

    Each evidence dict needs 'chunk_id' and 'chunk_text'; position in the
    list defines its citation number (1-based).
    """
    cited = extract_citations(sentence)
    claim = strip_citations(sentence)

    base = {
        "sentence": claim,
        "cited": cited,
        "best_entailment": 0.0,
        "best_contradiction": 0.0,
        "supporting_chunk": None,
        "contradicting_chunk": None,
        "abstention": is_abstention(claim),
    }

    if is_non_claim(claim) or len(claim.split()) < 4:
        return {**base, "label": "nei"}

    results = []
    for position, item in enumerate(evidence, start=1):
        ent, _, con = _nli_scores(str(item["chunk_text"]), claim)
        results.append({
            "position": position,
            "chunk_id": item.get("chunk_id"),
            "entailment": ent,
            "contradiction": con,
            "was_cited": position in cited,
        })

    if not results:
        return {**base, "label": "unsupported"}

    best_ent = max(results, key=lambda r: r["entailment"])
    best_con = max(results, key=lambda r: r["contradiction"])

    base.update({
        "best_entailment": round(best_ent["entailment"], 4),
        "best_contradiction": round(best_con["contradiction"], 4),
        "supporting_chunk": best_ent["chunk_id"],
        "contradicting_chunk": best_con["chunk_id"],
    })

    cited_results = [r for r in results if r["was_cited"]]
    cited_entails = any(
        r["entailment"] >= ENTAILMENT_THRESHOLD for r in cited_results
    )

    if cited_entails:
        label = "supported"
    elif best_con["contradiction"] >= CONTRADICTION_THRESHOLD:
        label = "contradicted"
    elif best_ent["entailment"] >= ENTAILMENT_THRESHOLD:
        label = "supported_uncited"
    else:
        label = "unsupported"

    return {**base, "label": label}


def summarise_labels(
    labels: list[str],
    any_citation: bool = True,
    n_abstention: int = 0,
) -> dict:
    """
    Compute the summary metrics from a list of per-sentence labels.

    Separated from verify_answer so a summary can be rebuilt from a saved
    claims file without re-running the NLI model.

    faithfulness, unsupported_claim_rate and citation_accuracy are None
    (not 0.0) where they are undefined: a rate over zero checkable claims is
    not a rate of zero, and a citation accuracy for an answer that cites
    nothing is not an accuracy of zero. Returning None lets pandas skip these
    rows in a mean instead of silently pulling it toward zero.
    """
    counts = {
        label: sum(1 for lab in labels if lab == label)
        for label in ["supported", "supported_uncited", "contradicted",
                      "unsupported", "nei"]
    }

    n_checkable = len(labels) - counts["nei"]
    grounded = counts["supported"] + counts["supported_uncited"]
    ungrounded = counts["unsupported"] + counts["contradicted"]

    if n_checkable > 0:
        faithfulness = round(grounded / n_checkable, 4)
        unsupported_rate = round(ungrounded / n_checkable, 4)
    else:
        faithfulness = None
        unsupported_rate = None

    if not any_citation:
        # The answer contains no citation markers at all, so citation accuracy
        # is undefined. This is the S0 case: reporting 0.0 would imply the
        # system cited badly rather than not at all.
        citation_accuracy = None
    elif grounded > 0:
        citation_accuracy = round(counts["supported"] / grounded, 4)
    else:
        citation_accuracy = 0.0

    if n_checkable == 0:
        # No verifier signal exists. Distinguished from LOW risk, which would
        # wrongly present an empty or abstaining answer as well grounded.
        risk = "ABSTAINED" if n_abstention > 0 else "NO_CLAIMS"
    elif counts["contradicted"] > 0 or unsupported_rate > RISK_HIGH_RATE:
        risk = "HIGH"
    elif unsupported_rate > RISK_MEDIUM_RATE:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "n_sentences": len(labels),
        "n_checkable": n_checkable,
        "n_abstention": n_abstention,
        **{f"n_{k}": v for k, v in counts.items()},
        "faithfulness": faithfulness,
        "unsupported_claim_rate": unsupported_rate,
        "citation_accuracy": citation_accuracy,
        "hallucination_risk": risk,
    }


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    """Verify a full answer. Returns per-sentence results plus summary metrics."""
    sentences = split_sentences(answer)
    claims = [verify_sentence(s, evidence) for s in sentences]

    summary = summarise_labels(
        [c["label"] for c in claims],
        any_citation=any(c["cited"] for c in claims),
        n_abstention=sum(1 for c in claims if c["abstention"]),
    )

    return {"claims": claims, **summary}


if __name__ == "__main__":
    evidence = [
        {"chunk_id": "demo_p1_c1",
         "chunk_text": "Students may work up to 20 hours per week during term time."},
        {"chunk_id": "demo_p2_c1",
         "chunk_text": "Tuition fees must be paid by the published instalment dates."},
    ]
    answer = (
        "Students may work up to 20 hours per week during term time [1]. "
        "Students may work up to 40 hours per week [1]. "
        "Fees are due by the published instalment dates. "
        "The University provides free legal representation to all students."
    )
    result = verify_answer(answer, evidence)
    for claim in result["claims"]:
        print(f"[{claim['label']:<18}] ent={claim['best_entailment']:.3f} "
              f"con={claim['best_contradiction']:.3f} :: {claim['sentence'][:70]}")
    print(f"\nFaithfulness:      {result['faithfulness']}")
    print(f"Unsupported rate:  {result['unsupported_claim_rate']}")
    print(f"Citation accuracy: {result['citation_accuracy']}")
    print(f"Risk:              {result['hallucination_risk']}")

    # Abstention case: the refusal template must yield no checkable claims.
    refusal = (
        "I cannot answer this question reliably from the available University "
        "documents. The retrieved evidence does not adequately support a "
        "complete answer: 1 claim(s) in the draft answer were contradicted by "
        "the retrieved evidence. Please consult the relevant policy document "
        "directly, or contact Student Services for authoritative guidance."
    )
    ref = verify_answer(refusal, evidence)
    print(
        f"\nAbstention check -> n_sentences={ref['n_sentences']} "
        f"n_checkable={ref['n_checkable']} n_abstention={ref['n_abstention']} "
        f"faithfulness={ref['faithfulness']} risk={ref['hallucination_risk']}"
    )
    assert ref["n_checkable"] == 0, "refusal template still produces claims"
    assert ref["faithfulness"] is None, "refusal must not score 0.0"