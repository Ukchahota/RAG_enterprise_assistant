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
"""

import re
from functools import lru_cache

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

ENTAILMENT_THRESHOLD = 0.50
CONTRADICTION_THRESHOLD = 0.50

# Sentences that assert nothing checkable.
NON_CLAIM_PATTERNS = [
    r"^\s*(here|below|the following|in summary|to summari[sz]e)\b",
    r"^\s*(i hope|let me know|feel free|please note that if)\b",
    r"^\s*(the evidence (does not|doesn't) (contain|include|provide))",
    r"^\s*(based on the (provided )?evidence)[,:]?\s*$",
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

    parts = re.split(r"(?<=[.!?])\s+", protected)
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


def is_non_claim(sentence: str) -> bool:
    low = sentence.lower()
    return any(re.search(p, low) for p in NON_CLAIM_PATTERNS)


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


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    """Verify a full answer. Returns per-sentence results plus summary metrics."""
    sentences = split_sentences(answer)
    claims = [verify_sentence(s, evidence) for s in sentences]

    checkable = [c for c in claims if c["label"] != "nei"]
    n = len(checkable) or 1

    counts = {
        label: sum(1 for c in claims if c["label"] == label)
        for label in ["supported", "supported_uncited", "contradicted",
                      "unsupported", "nei"]
    }

    grounded = counts["supported"] + counts["supported_uncited"]
    faithfulness = grounded / n
    unsupported_rate = (counts["unsupported"] + counts["contradicted"]) / n
    citation_accuracy = counts["supported"] / (grounded or 1)

    if counts["contradicted"] > 0 or unsupported_rate > 0.5:
        risk = "HIGH"
    elif unsupported_rate > 0.2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "claims": claims,
        "n_sentences": len(claims),
        "n_checkable": len(checkable),
        **{f"n_{k}": v for k, v in counts.items()},
        "faithfulness": round(faithfulness, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "citation_accuracy": round(citation_accuracy, 4),
        "hallucination_risk": risk,
    }


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