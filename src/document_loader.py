from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

import fitz  # PyMuPDF
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw_documents"

INVENTORY_PATH = DATA_DIR / "document_inventory.csv"
OUTPUT_PATH = DATA_DIR / "extracted_pages.csv"


# ============================================================
# INVENTORY SCHEMA
# ============================================================

REQUIRED_COLUMNS = [
    "document_name",
    "file_name",
]

OPTIONAL_COLUMNS = [
    "document_type",
    "source",
    "source_url",
    "date_accessed",
    "used_in_rag",
    "notes",
]

OUTPUT_COLUMNS = [
    "document_name",
    "document_type",
    "source",
    "source_url",
    "file_name",
    "file_path",
    "date_accessed",
    "used_in_rag",
    "notes",
    "page_number",
    "total_pages",
    "text",
    "page_text",
    "character_count",
    "word_count",
    "has_text",
]


# Alternative inventory column names.
#
# Important:
# file_name is not renamed to source.
# source and file_name are separate pieces of metadata.
COLUMN_ALIASES = {
    # Document name
    "document": "document_name",
    "document_title": "document_name",
    "title": "document_name",
    "name": "document_name",

    # Document type
    "type": "document_type",
    "category": "document_type",
    "document_category": "document_type",

    # Local PDF filename
    "filename": "file_name",
    "file": "file_name",
    "pdf": "file_name",
    "pdf_file": "file_name",
    "source_file": "file_name",
    "local_file": "file_name",

    # Source organisation or publisher
    "publisher": "source",
    "organisation": "source",
    "organization": "source",

    # Source URL
    "url": "source_url",
    "link": "source_url",
    "document_url": "source_url",

    # Date accessed
    "accessed": "date_accessed",
    "access_date": "date_accessed",

    # Used in RAG
    "include": "used_in_rag",
    "included": "used_in_rag",
    "use_in_rag": "used_in_rag",
    "active": "used_in_rag",

    # Notes
    "description": "notes",
    "comment": "notes",
    "comments": "notes",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalise_column_name(column: Any) -> str:
    """
    Convert a column name to lowercase snake_case.

    Examples:
        "Document Name" -> "document_name"
        "Source URL"    -> "source_url"
        "File-Name"     -> "file_name"
    """
    column_name = str(column).strip().lower()

    column_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        column_name,
    )

    column_name = re.sub(
        r"_+",
        "_",
        column_name,
    )

    return column_name.strip("_")


def clean_cell_value(value: Any) -> str:
    """
    Convert an inventory value to a clean string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def is_enabled_for_rag(value: Any) -> bool:
    """
    Interpret the used_in_rag value.

    Values treated as True:
        yes, y, true, 1, include, included, active

    Values treated as False:
        no, n, false, 0, exclude, excluded, inactive

    Empty values default to True.
    """
    cleaned_value = clean_cell_value(value).lower()

    if cleaned_value == "":
        return True

    true_values = {
        "yes",
        "y",
        "true",
        "1",
        "include",
        "included",
        "active",
    }

    false_values = {
        "no",
        "n",
        "false",
        "0",
        "exclude",
        "excluded",
        "inactive",
    }

    if cleaned_value in true_values:
        return True

    if cleaned_value in false_values:
        return False

    print(
        f"Warning: unknown used_in_rag value "
        f"'{value}'. The document will be included."
    )

    return True


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: Any) -> str:
    """
    Clean text extracted from a PDF page.

    The cleaning is conservative so important policy wording,
    punctuation and paragraph content are preserved.
    """
    if not isinstance(text, str):
        return ""

    # Remove null characters.
    text = text.replace("\x00", " ")

    # Remove soft hyphens.
    text = text.replace("\u00ad", "")

    # Standardise line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words split across lines.
    #
    # Example:
    # "regula-\ntions" becomes "regulations"
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    # Preserve paragraph boundaries temporarily.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Convert single line breaks into spaces.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Convert remaining paragraph breaks into spaces.
    text = re.sub(
        r"\n+",
        " ",
        text,
    )

    # Collapse repeated spaces and tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# INVENTORY COLUMN NORMALISATION
# ============================================================

def normalise_inventory_columns(
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise inventory column names and apply safe aliases.

    The function avoids creating duplicate columns. For example,
    if both source and file_name already exist, file_name will not
    be renamed to source.
    """
    inventory_df = inventory_df.copy()

    inventory_df.columns = [
        normalise_column_name(column)
        for column in inventory_df.columns
    ]

    duplicate_columns_before_aliases = (
        inventory_df.columns[
            inventory_df.columns.duplicated()
        ].tolist()
    )

    if duplicate_columns_before_aliases:
        raise ValueError(
            "Duplicate inventory columns found after header "
            f"normalisation: {duplicate_columns_before_aliases}. "
            f"All columns: {inventory_df.columns.tolist()}"
        )

    rename_mapping: dict[str, str] = {}

    existing_columns = set(inventory_df.columns)

    for original_column in inventory_df.columns:
        target_column = COLUMN_ALIASES.get(original_column)

        if target_column is None:
            continue

        if target_column == original_column:
            continue

        if target_column in existing_columns:
            print(
                f"Warning: keeping '{original_column}' unchanged "
                f"because '{target_column}' already exists."
            )
            continue

        if target_column in rename_mapping.values():
            print(
                f"Warning: keeping '{original_column}' unchanged "
                f"because another column is already being renamed "
                f"to '{target_column}'."
            )
            continue

        rename_mapping[original_column] = target_column

    if rename_mapping:
        inventory_df = inventory_df.rename(
            columns=rename_mapping
        )

    duplicate_columns_after_aliases = (
        inventory_df.columns[
            inventory_df.columns.duplicated()
        ].tolist()
    )

    if duplicate_columns_after_aliases:
        raise ValueError(
            "Duplicate inventory columns found after aliases "
            f"were applied: {duplicate_columns_after_aliases}. "
            f"All columns: {inventory_df.columns.tolist()}"
        )

    return inventory_df


# ============================================================
# INVENTORY CREATION
# ============================================================

def find_pdf_files() -> list[Path]:
    """
    Find all PDF files inside data/raw_documents.

    Subdirectories are also searched.
    """
    RAW_DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        path
        for path in RAW_DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    )


def create_inventory_from_pdf_folder() -> pd.DataFrame:
    """
    Create document_inventory.csv using PDFs found inside
    data/raw_documents.
    """
    pdf_files = find_pdf_files()

    inventory_rows: list[dict[str, str]] = []

    for pdf_path in pdf_files:
        relative_path = pdf_path.relative_to(
            RAW_DOCUMENTS_DIR
        )

        inventory_rows.append(
            {
                "document_name": pdf_path.stem,
                "document_type": "",
                "source": "",
                "source_url": "",
                "file_name": str(relative_path),
                "date_accessed": "",
                "used_in_rag": "yes",
                "notes": "",
            }
        )

    inventory_columns = (
        REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    )

    # Remove duplicates while preserving order.
    inventory_columns = list(
        dict.fromkeys(inventory_columns)
    )

    inventory_df = pd.DataFrame(
        inventory_rows,
        columns=inventory_columns,
    )

    INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_df.to_csv(
        INVENTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Created inventory: {INVENTORY_PATH}")
    print(f"PDF files added: {len(inventory_df)}")

    return inventory_df


# ============================================================
# INVENTORY LOADING AND VALIDATION
# ============================================================

def load_inventory() -> pd.DataFrame:
    """
    Load, normalise and validate document_inventory.csv.
    """
    if not INVENTORY_PATH.exists():
        print(
            f"Inventory file not found: {INVENTORY_PATH}"
        )
        print(
            "Creating an inventory from PDFs in "
            "data/raw_documents..."
        )

        return create_inventory_from_pdf_folder()

    try:
        inventory_df = pd.read_csv(
            INVENTORY_PATH,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"The inventory file is empty: {INVENTORY_PATH}"
        ) from error

    except UnicodeDecodeError:
        try:
            inventory_df = pd.read_csv(
                INVENTORY_PATH,
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
            )

        except Exception as error:
            raise RuntimeError(
                "The inventory CSV could not be decoded. "
                f"File: {INVENTORY_PATH}"
            ) from error

    except Exception as error:
        raise RuntimeError(
            f"Could not read inventory: {INVENTORY_PATH}"
        ) from error

    print(
        "Original inventory columns:",
        inventory_df.columns.tolist(),
    )

    inventory_df = normalise_inventory_columns(
        inventory_df
    )

    print(
        "Normalised inventory columns:",
        inventory_df.columns.tolist(),
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in inventory_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nInventory validation failed.\n"
            f"Missing required columns: {missing_columns}\n"
            f"Detected columns: {inventory_df.columns.tolist()}\n"
            f"Inventory file: {INVENTORY_PATH}\n\n"
            "The required columns are:\n"
            "document_name,file_name"
        )

    # Create any optional columns that are not present.
    for column in OPTIONAL_COLUMNS:
        if column not in inventory_df.columns:
            inventory_df[column] = ""

    columns_to_keep = (
        REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    )

    # Remove duplicate names while preserving order.
    columns_to_keep = list(
        dict.fromkeys(columns_to_keep)
    )

    inventory_df = inventory_df[
        columns_to_keep
    ].copy()

    # Clean every inventory column.
    for column in inventory_df.columns:
        inventory_df[column] = (
            inventory_df[column]
            .map(clean_cell_value)
        )

    # Remove completely blank rows.
    blank_rows = (
        inventory_df
        .apply(
            lambda row: all(
                value == ""
                for value in row
            ),
            axis=1,
        )
    )

    if blank_rows.any():
        blank_count = int(blank_rows.sum())

        print(
            f"Warning: removing {blank_count} completely "
            "blank inventory row(s)."
        )

        inventory_df = inventory_df.loc[
            ~blank_rows
        ].copy()

    # Remove rows with no filename.
    empty_file_rows = (
        inventory_df["file_name"] == ""
    )

    if empty_file_rows.any():
        empty_file_count = int(
            empty_file_rows.sum()
        )

        print(
            f"Warning: removing {empty_file_count} row(s) "
            "with an empty file_name."
        )

        inventory_df = inventory_df.loc[
            ~empty_file_rows
        ].copy()

    # Fill missing document names using the filename.
    missing_document_names = (
        inventory_df["document_name"] == ""
    )

    if missing_document_names.any():
        inventory_df.loc[
            missing_document_names,
            "document_name",
        ] = inventory_df.loc[
            missing_document_names,
            "file_name",
        ].map(
            lambda value: Path(value).stem
        )

    # Convert used_in_rag to a consistent yes/no value.
    inventory_df["used_in_rag"] = (
        inventory_df["used_in_rag"]
        .map(
            lambda value: (
                "yes"
                if is_enabled_for_rag(value)
                else "no"
            )
        )
    )

    # Keep only active RAG documents.
    disabled_rows = (
        inventory_df["used_in_rag"] == "no"
    )

    if disabled_rows.any():
        disabled_count = int(
            disabled_rows.sum()
        )

        print(
            f"Skipping {disabled_count} document(s) marked "
            "as not used in RAG."
        )

        inventory_df = inventory_df.loc[
            ~disabled_rows
        ].copy()

    # Warn about duplicate file entries.
    duplicated_files = inventory_df[
        inventory_df["file_name"].duplicated(
            keep=False
        )
    ]

    if not duplicated_files.empty:
        duplicate_names = (
            duplicated_files["file_name"]
            .drop_duplicates()
            .tolist()
        )

        print(
            "Warning: duplicate file_name entries found: "
            f"{duplicate_names}"
        )

    return inventory_df.reset_index(drop=True)


# ============================================================
# PDF PATH RESOLUTION
# ============================================================

def resolve_pdf_path(file_name: str) -> Path:
    """
    Resolve the file_name value to a local PDF path.

    Supported values include:

        Payment Policy.pdf

        policies/Payment Policy.pdf

        data/raw_documents/Payment Policy.pdf

        C:/Users/example/document.pdf
    """
    cleaned_file_name = clean_cell_value(file_name)

    if not cleaned_file_name:
        return RAW_DOCUMENTS_DIR

    file_path = Path(cleaned_file_name)

    # Absolute path.
    if file_path.is_absolute():
        return file_path

    candidate_paths = [
        RAW_DOCUMENTS_DIR / file_path,
        PROJECT_ROOT / file_path,
        DATA_DIR / file_path,
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    # Case-insensitive filename search.
    requested_name = file_path.name.lower()

    matching_files = [
        path
        for path in RAW_DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.name.lower() == requested_name
    ]

    if len(matching_files) == 1:
        return matching_files[0].resolve()

    if len(matching_files) > 1:
        print(
            f"Warning: multiple PDFs match '{file_name}'. "
            f"Using: {matching_files[0]}"
        )

        return matching_files[0].resolve()

    # Return the expected path for a useful warning.
    return (
        RAW_DOCUMENTS_DIR / file_path
    ).resolve()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_page_text(
    page: fitz.Page,
) -> str:
    """
    Extract text from one PDF page.
    """
    raw_text = page.get_text(
        "text",
        sort=True,
    )

    return clean_text(raw_text)


def extract_pdf_pages(
    pdf_path: Path,
    inventory_row: pd.Series,
) -> list[dict[str, Any]]:
    """
    Extract all pages from one PDF.

    One output dictionary is produced for every page.
    """
    extracted_rows: list[dict[str, Any]] = []

    document_name = clean_cell_value(
        inventory_row.get(
            "document_name",
            pdf_path.stem,
        )
    )

    document_type = clean_cell_value(
        inventory_row.get(
            "document_type",
            "",
        )
    )

    source = clean_cell_value(
        inventory_row.get(
            "source",
            "",
        )
    )

    source_url = clean_cell_value(
        inventory_row.get(
            "source_url",
            "",
        )
    )

    file_name = clean_cell_value(
        inventory_row.get(
            "file_name",
            pdf_path.name,
        )
    )

    date_accessed = clean_cell_value(
        inventory_row.get(
            "date_accessed",
            "",
        )
    )

    used_in_rag = clean_cell_value(
        inventory_row.get(
            "used_in_rag",
            "yes",
        )
    )

    notes = clean_cell_value(
        inventory_row.get(
            "notes",
            "",
        )
    )

    try:
        pdf_document = fitz.open(pdf_path)

    except Exception as error:
        print(f"Error opening PDF: {pdf_path}")
        print(f"Reason: {error}")

        return extracted_rows

    try:
        if pdf_document.needs_pass:
            print(
                f"Warning: password-protected PDF skipped: "
                f"{pdf_path.name}"
            )

            return extracted_rows

        total_pages = pdf_document.page_count

        for page_index in range(total_pages):
            page_number = page_index + 1

            try:
                page = pdf_document.load_page(
                    page_index
                )

                cleaned_text = extract_page_text(
                    page
                )

            except Exception as error:
                print(
                    f"Warning: failed to extract "
                    f"{pdf_path.name}, page {page_number}: "
                    f"{error}"
                )

                cleaned_text = ""

            word_count = (
                len(cleaned_text.split())
                if cleaned_text
                else 0
            )

            extracted_rows.append(
                {
                    "document_name": document_name,
                    "document_type": document_type,
                    "source": source,
                    "source_url": source_url,
                    "file_name": file_name,
                    "file_path": str(pdf_path),
                    "date_accessed": date_accessed,
                    "used_in_rag": used_in_rag,
                    "notes": notes,
                    "page_number": page_number,
                    "total_pages": total_pages,

                    # Both columns are included for compatibility
                    # with different chunking scripts.
                    "text": cleaned_text,
                    "page_text": cleaned_text,

                    "character_count": len(cleaned_text),
                    "word_count": word_count,
                    "has_text": bool(cleaned_text),
                }
            )

    finally:
        pdf_document.close()

    return extracted_rows


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_extracted_pages(
    extracted_df: pd.DataFrame,
) -> None:
    """
    Print quality statistics for the extraction output.
    """
    if extracted_df.empty:
        print("No PDF pages were extracted.")
        return

    total_documents = int(
        extracted_df["document_name"].nunique()
    )

    total_pages = len(extracted_df)

    pages_with_text = int(
        extracted_df["has_text"].sum()
    )

    empty_pages = (
        total_pages - pages_with_text
    )

    total_words = int(
        extracted_df["word_count"].sum()
    )

    total_characters = int(
        extracted_df["character_count"].sum()
    )

    average_words_per_page = (
        total_words / total_pages
        if total_pages
        else 0
    )

    print("\nExtraction summary")
    print("=" * 60)
    print(f"Documents processed: {total_documents}")
    print(f"Total pages extracted: {total_pages}")
    print(f"Pages containing text: {pages_with_text}")
    print(f"Empty or scanned pages: {empty_pages}")
    print(f"Total extracted words: {total_words:,}")
    print(
        f"Total extracted characters: "
        f"{total_characters:,}"
    )
    print(
        f"Average words per page: "
        f"{average_words_per_page:.2f}"
    )

    if total_pages > 0:
        percentage_with_text = (
            pages_with_text / total_pages
        ) * 100

        print(
            f"Pages with extractable text: "
            f"{percentage_with_text:.2f}%"
        )

    if empty_pages > 0:
        empty_page_details = extracted_df.loc[
            ~extracted_df["has_text"],
            [
                "document_name",
                "page_number",
            ],
        ]

        print(
            "\nWarning: some pages contain no extractable "
            "text. They may be scanned images."
        )

        print(
            empty_page_details
            .head(20)
            .to_string(index=False)
        )

        if len(empty_page_details) > 20:
            print(
                f"...and {len(empty_page_details) - 20} "
                "more empty pages."
            )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_extracted_pages(
    extracted_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Create, sort and save the extracted pages DataFrame.
    """
    extracted_df = pd.DataFrame(
        extracted_rows,
        columns=OUTPUT_COLUMNS,
    )

    if extracted_df.empty:
        raise RuntimeError(
            "No document content was extracted."
        )

    extracted_df = extracted_df.sort_values(
        by=[
            "document_name",
            "page_number",
        ],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return extracted_df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Run the complete document-loading pipeline.
    """
    print("=" * 60)
    print("RAG Document Loader")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw documents: {RAW_DOCUMENTS_DIR}")
    print(f"Inventory: {INVENTORY_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    RAW_DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_df = load_inventory()

    if inventory_df.empty:
        raise ValueError(
            "No active documents were found in the inventory.\n"
            "Check that:\n"
            "1. file_name contains valid PDF filenames\n"
            "2. used_in_rag is not set to no\n"
            f"3. PDFs exist inside {RAW_DOCUMENTS_DIR}"
        )

    print(
        f"\nActive inventory documents: "
        f"{len(inventory_df)}"
    )

    all_extracted_rows: list[
        dict[str, Any]
    ] = []

    successful_documents = 0
    missing_documents = 0
    failed_documents = 0
    unsupported_documents = 0

    for row_index, inventory_row in (
        inventory_df.iterrows()
    ):
        document_number = row_index + 1
        total_documents = len(inventory_df)

        document_name = clean_cell_value(
            inventory_row["document_name"]
        )

        file_name = clean_cell_value(
            inventory_row["file_name"]
        )

        pdf_path = resolve_pdf_path(
            file_name
        )

        print(
            f"\n[{document_number}/{total_documents}] "
            f"Processing: {document_name}"
        )

        print(f"Inventory filename: {file_name}")
        print(f"Resolved PDF path: {pdf_path}")

        if not pdf_path.exists():
            print(
                f"Warning: PDF file not found: {pdf_path}"
            )

            missing_documents += 1
            continue

        if not pdf_path.is_file():
            print(
                f"Warning: path is not a file: {pdf_path}"
            )

            failed_documents += 1
            continue

        if pdf_path.suffix.lower() != ".pdf":
            print(
                "Warning: unsupported file type: "
                f"{pdf_path.suffix}"
            )

            unsupported_documents += 1
            continue

        extracted_rows = extract_pdf_pages(
            pdf_path=pdf_path,
            inventory_row=inventory_row,
        )

        if not extracted_rows:
            print(
                f"Warning: no pages extracted from "
                f"{pdf_path.name}"
            )

            failed_documents += 1
            continue

        all_extracted_rows.extend(
            extracted_rows
        )

        successful_documents += 1

        pages_with_text = sum(
            1
            for row in extracted_rows
            if row["has_text"]
        )

        print(
            f"Extracted {len(extracted_rows)} pages "
            f"({pages_with_text} containing text)."
        )

    if not all_extracted_rows:
        raise RuntimeError(
            "\nNo document content was extracted.\n\n"
            "Check the following:\n"
            f"1. PDFs exist in: {RAW_DOCUMENTS_DIR}\n"
            "2. Inventory file_name values match the PDF "
            "filenames\n"
            "3. Documents are valid PDF files\n"
            "4. used_in_rag is set to yes\n"
            "5. PDFs are not password protected"
        )

    extracted_df = save_extracted_pages(
        all_extracted_rows
    )

    validate_extracted_pages(
        extracted_df
    )

    print("\nDocument processing status")
    print("=" * 60)
    print(
        f"Successfully processed: "
        f"{successful_documents}"
    )
    print(
        f"Missing documents: {missing_documents}"
    )
    print(
        f"Failed documents: {failed_documents}"
    )
    print(
        f"Unsupported documents: "
        f"{unsupported_documents}"
    )

    print(
        "\nExtracted page data saved to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nDocument loading was cancelled.")
        sys.exit(1)

    except Exception as error:
        print("\nDocument loader failed.")
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(f"Error message: {error}")

        sys.exit(1)