from src.faithfulness import _nli_scores
from src.evaluate_hallucination import build_evidence, load_chunk_texts
import pandas as pd

claim = ("After a student is notified that programme-fee payments remain outstanding, "
         "the University will give them 5 working days to settle the outstanding balance.")

a = pd.read_csv(r"results\answers_s3_corrective_rag.csv")
row = a[a["question_id"] == "V2Q008"].iloc[0]
full = build_evidence(row, load_chunk_texts())[0]["chunk_text"]

for n in (200, 300, 400, 500, 600, 700, 800):
    ent, neu, con = _nli_scores(full[:n], claim)
    print(f"premise[:{n:4d}]  ent={ent:.3f} neu={neu:.3f} con={con:.3f}")
