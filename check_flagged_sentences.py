import pandas as pd

c = pd.read_csv(r"results\hallucination_claims_s1_s3.csv")
c = c[(c["system"] == "S3_corrective_rag")
      & (c["question_id"].isin(["V2Q075", "V2Q008", "V2Q038"]))]

for qid, g in c.groupby("question_id"):
    print("=" * 70)
    print(qid)
    for _, r in g.iterrows():
        print(f"  [{r['label']}] ent={r['best_entailment']:.3f} con={r['best_contradiction']:.3f} cited={r['cited']}")
        print(f"      {str(r['sentence'])[:220]}")
