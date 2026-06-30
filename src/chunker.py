from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed_chunks" / "extracted_pages.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_chunks"
OUTPUT_PATH = OUTPUT_DIR / "document_chunks.csv"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """
    Basic text cleaning for extracted PDF text.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Simple method for first RAG baseline.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages_df = pd.read_csv(INPUT_PATH)

    rows = []

    for _, row in tqdm(pages_df.iterrows(), total=len(pages_df)):
        cleaned_text = clean_text(row["text"])

        chunks = split_text_into_chunks(
            cleaned_text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_id = (
                f"{row['file_name']}_p{row['page_number']}_c{chunk_index}"
            )

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": row["document_name"],
                    "document_type": row["document_type"],
                    "source": row["source"],
                    "source_url": row["source_url"],
                    "file_name": row["file_name"],
                    "date_accessed": row["date_accessed"],
                    "page_number": row["page_number"],
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text),
                }
            )

    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Created {len(chunks_df)} chunks.")
    print(f"Saved chunks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()