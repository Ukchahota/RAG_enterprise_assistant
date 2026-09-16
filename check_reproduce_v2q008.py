import pandas as pd
from src.faithfulness import verify_sentence, strip_citations
from src.evaluate_hallucination import build_evidence, load_chunk_texts

a = pd.read_csv(r"results\answers_s3_corrective_rag.csv")
row = a[a["question_id"] == "V2Q008"].iloc[0]

ct = load_chunk_texts()
ev = build_evidence(row, ct)
print("n evidence:", len(ev))
for i, e in enumerate(ev, 1):
    print(f"  [{i}] {e['chunk_id']}  len={len(e['chunk_text'])}  starts: {e['chunk_text'][:60]!r}")

c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
sent = c[(c["system"] == "S3_corrective_rag") & (c["question_id"] == "V2Q008")
         & (c["label"] == "unsupported")]["sentence"].iloc[0]
print("\nclaim:", repr(strip_citations(str(sent))))

res = verify_sentence(str(sent), ev)
print("\nlabel:", res["label"], "best_ent:", res["best_entailment"],
      "supporting:", res["supporting_chunk"], "cited:", res["cited"])
