# label_claims.py
import pandas as pd, os, textwrap

BLIND = r"results\relabel_batch.csv"
OUT   = r"results\detector_validation_human_pass2.csv"

LABELS = {"1": "supported", "2": "supported_uncited", "3": "contradicted",
          "4": "unsupported", "5": "nei","6":"invalid"}

df = pd.read_csv(BLIND)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["row_id", "human_label"])
seen = set(done["row_id"])

for _, r in df.iterrows():
    if r["row_id"] in seen:
        continue
    print("\n" + "=" * 78)
    print(f"row {r['row_id']}   |   cited: {r.get('cited', 'NaN')}")
    print("-" * 78)
    print("CLAIM:\n" + textwrap.fill(str(r["sentence"]), 76))
    print("-" * 78)
    chunk = str(r.get("supporting_chunk_text", "nan"))
    print("EVIDENCE:\n" + (textwrap.fill(chunk[:1200], 76) if chunk != "nan" else "(none retrieved)"))
    print("=" * 78)
    print("1 supported  2 supported_uncited  3 contradicted  4 unsupported  5 nei  |  q quit")

    while True:
        k = input("> ").strip().lower()
        if k == "q":
            pd.DataFrame(done).to_csv(OUT, index=False)
            print(f"\nSaved {len(done)}/{len(df)}. Rerun to resume.")
            raise SystemExit
        if k in LABELS:
            done = pd.concat([done, pd.DataFrame([{"row_id": r["row_id"], "human_label": LABELS[k]}])], ignore_index=True)
            done.to_csv(OUT, index=False)
            break
        print("1-5 or q")

print(f"\nAll {len(df)} labelled -> {OUT}")