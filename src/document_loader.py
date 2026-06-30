from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_PATH = PROJECT_ROOT / "data" / "document_inventory.csv"
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_documents"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_chunks"
OUTPUT_PATH = OUTPUT_DIR / "extracted_pages.csv"


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text page by page from a PDF file.
    Returns a list of dictionaries, one per page.
    """
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "page_number": page_index,
                    "text": text,
                }
            )

    return pages


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Document inventory not found: {INVENTORY_PATH}")

    inventory = pd.read_csv(INVENTORY_PATH)

    required_columns = [
        "document_name",
        "document_type",
        "source",
        "source_url",
        "file_name",
        "date_accessed",
        "used_in_rag",
        "notes",
    ]

    missing_columns = [col for col in required_columns if col not in inventory.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in inventory: {missing_columns}")

    rows = []

    inventory = inventory[inventory["used_in_rag"].astype(str).str.lower() == "yes"]

    for _, doc_row in tqdm(inventory.iterrows(), total=len(inventory)):
        file_name = doc_row["file_name"]
        pdf_path = RAW_DOCS_DIR / file_name

        if not pdf_path.exists():
            print(f"WARNING: File not found: {pdf_path}")
            continue

        pages = extract_pdf_pages(pdf_path)

        for page in pages:
            rows.append(
                {
                    "document_name": doc_row["document_name"],
                    "document_type": doc_row["document_type"],
                    "source": doc_row["source"],
                    "source_url": doc_row["source_url"],
                    "file_name": file_name,
                    "date_accessed": doc_row["date_accessed"],
                    "page_number": page["page_number"],
                    "text": page["text"],
                }
            )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Extracted {len(output_df)} pages.")
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()