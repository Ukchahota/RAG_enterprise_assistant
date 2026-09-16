import pandas as pd
from src.evaluate_hallucination import build_evidence, load_chunk_texts, _split_ids

ct = load_chunk_texts()
a = pd.read_csv(r"results\answers_s0_llm_only.csv")

hit = miss = 0
examples = []
for _, r in a.iterrows():
    ids = _split_ids(r.get("gold_chunk_ids"))
    for i in ids:
        if i in ct:
            hit += 1
        else:
            miss += 1
            if len(examples) < 5:
                examples.append(i)

print(f"gold ids found in index: {hit}")
print(f"gold ids missing:        {miss}")
print("missing examples:", examples)
print()
ev = build_evidence(a.iloc[0], ct)
print("evidence for row 0:", len(ev), "chunks")
print("row 0 gold ids:", _split_ids(a.iloc[0].get("gold_chunk_ids")))
print("row 0 evidence_chunk_ids:", repr(a.iloc[0].get("evidence_chunk_ids")))
