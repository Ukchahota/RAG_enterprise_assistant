# score_correctness.py
# Manual correctness scoring on a stratified subset of the common question set.
import pandas as pd, os, textwrap

COMMON = r"results\common_question_set.csv"
S3     = r"results\answers_s3_corrective_rag.csv"
SAMPLE = r"results\correctness_sample.csv"
OUT    = r"results\correctness_scores.csv"

LABELS = {"1": "correct", "2": "partial", "3": "incorrect", "4": "refused"}

if not os.path.exists(SAMPLE):
    common = pd.read_csv(COMMON)["question_id"]
    d = pd.read_csv(S3)
    d = d[d["question_id"].isin(common)].copy()

    frac = 30 / len(d)
    idx = []
    for qt, grp in d.groupby("question_type"):
        n = max(2, round(len(grp) * frac))
        idx.extend(grp.sample(min(n, len(grp)), random_state=42).index.tolist())

    samp = d.loc[idx].head(30)[["question_id", "question_type", "question",
                                "expected_answer", "answer", "corrective_action"]]
    samp.to_csv(SAMPLE, index=False)
    print(f"Sampled {len(samp)} questions -> {SAMPLE}")
    print(samp["question_type"].value_counts().to_string())
    print()

samp = pd.read_csv(SAMPLE)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["question_id", "correctness"])
seen = set(done["question_id"])

for _, r in samp.iterrows():
    if r["question_id"] in seen:
        continue
    print("\n" + "=" * 78)
    print(f"{r['question_id']}  |  {r['question_type']}  |  {r['corrective_action']}")
    print("-" * 78)
    print("Q: " + textwrap.fill(str(r["question"]), 76))
    print("\nEXPECTED:\n" + textwrap.fill(str(r["expected_answer"])[:700], 76))
    print("\nSYSTEM:\n" + textwrap.fill(str(r["answer"])[:700], 76))
    print("=" * 78)
    print("1 correct   2 partially correct   3 incorrect   4 refused (n/a)   | q quit")

    while True:
        k = input("> ").strip().lower()
        if k == "q":
            print(f"\nSaved {len(done)}/{len(samp)}. Rerun to resume.")
            raise SystemExit
        if k in LABELS:
            done = pd.concat(
                [done, pd.DataFrame([{"question_id": r["question_id"], "correctness": LABELS[k]}])],
                ignore_index=True)
            done.to_csv(OUT, index=False)
            break
        print("1-4 or q")

print(f"\nAll {len(samp)} scored -> {OUT}")