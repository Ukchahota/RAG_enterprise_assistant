import pandas as pd, json

a = pd.read_csv(r"results\answers_s3_corrective_rag.csv")
r = a[a["question_id"] == "V2Q008"].iloc[0]
ids = [x.strip() for x in str(r["evidence_chunk_ids"]).split("|") if x.strip()]
try:
    texts = json.loads(r["evidence_texts"])
except Exception:
    texts = []
print("chunk 1 =", ids[0] if ids else "none")
print("-" * 70)
print(str(texts[0])[:1500] if texts else "evidence_texts not parseable")
