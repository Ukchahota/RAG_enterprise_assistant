# merge_and_score.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

key = pd.read_csv(r"results\detector_validation_key.csv").drop(columns=["human_label"])
p1  = pd.read_csv(r"results\detector_validation_human_pass1.csv")
p2  = pd.read_csv(r"results\detector_validation_human_pass2.csv")

# pass 2 overrides pass 1 for the rows it covers
final = pd.concat([p1[~p1["row_id"].isin(p2["row_id"])], p2], ignore_index=True)
final = final.sort_values("row_id").reset_index(drop=True)
final.to_csv(r"results\detector_validation_human_final.csv", index=False)

changed = p1.merge(p2, on="row_id", suffixes=("_p1", "_p2"))
changed = changed[changed["human_label_p1"] != changed["human_label_p2"]]
print(f"Rows revised under corrected guideline: {len(changed)}/{len(p2)}\n")

df = key[["row_id", "question_type", "predicted_label"]].merge(final, on="row_id")
labels = ["supported", "supported_uncited", "contradicted", "unsupported", "nei"]

print(f"Agreement: {(df['predicted_label'] == df['human_label']).mean():.3f}  (stratified sample)\n")
print("Per-class precision (the valid metric under prediction-stratified sampling):")
print(classification_report(df["human_label"], df["predicted_label"],
                            labels=labels, zero_division=0))

print("\nConfusion matrix (rows = human, cols = model):")
print(pd.DataFrame(confusion_matrix(df["human_label"], df["predicted_label"], labels=labels),
                   index=labels, columns=labels))

print("\nRemaining disagreements:")
d = df[df["predicted_label"] != df["human_label"]]
print(d[["row_id", "question_type", "predicted_label", "human_label"]].to_string(index=False))