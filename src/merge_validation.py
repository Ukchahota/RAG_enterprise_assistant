from pathlib import Path
import pandas as pd

blind = pd.read_csv("results/detector_validation_blind.csv", encoding="utf-8-sig")
key = pd.read_csv("results/detector_validation_key.csv", encoding="utf-8-sig")

merged = key.drop(columns=["human_label", "notes"], errors="ignore").merge(
    blind[["row_id", "human_label", "notes"]], on="row_id", how="left"
)
merged.to_csv("results/detector_validation_sample.csv",
              index=False, encoding="utf-8-sig")

done = merged["human_label"].astype(str).str.strip().ne("").sum()
print(f"Merged {done} labelled of {len(merged)}")
print("Now run: python -m src.evaluate_detector")
