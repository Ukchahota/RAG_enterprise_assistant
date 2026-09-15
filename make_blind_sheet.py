from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
src = Path("results/detector_validation_sample.csv")
df = pd.read_csv(src, encoding="utf-8-sig")

df = df.reset_index().rename(columns={"index": "row_id"})

# Blind sheet: no predicted_label, no NLI scores.
blind = df[[
    "row_id", "question_id", "sentence",
    "supporting_chunk", "supporting_chunk_text",
]].copy()
blind["human_label"] = ""
blind["notes"] = ""
blind.to_csv("results/detector_validation_blind.csv",
             index=False, encoding="utf-8-sig")

# Key: everything needed to score, merged back on row_id.
df.to_csv("results/detector_validation_key.csv",
          index=False, encoding="utf-8-sig")

print("Blind sheet:", len(blind), "rows -> results/detector_validation_blind.csv")
print("Key saved   -> results/detector_validation_key.csv")
print("\nLabel the blind sheet, then run: python -m src.merge_validation")
