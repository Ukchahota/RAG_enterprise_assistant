from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_chunks" / "document_chunks.csv"

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.index"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> pd.DataFrame:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks_df = pd.read_csv(CHUNKS_PATH)

    required_columns = [
        "chunk_id",
        "document_name",
        "document_type",
        "source",
        "source_url",
        "file_name",
        "page_number",
        "chunk_text",
    ]

    missing_columns = [col for col in required_columns if col not in chunks_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in chunks file: {missing_columns}")

    chunks_df = chunks_df.dropna(subset=["chunk_text"]).reset_index(drop=True)
    return chunks_df


def create_embeddings(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    return index


def main() -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    chunks_df = load_chunks()

    print(f"Loaded {len(chunks_df)} chunks.")
    print(f"Using embedding model: {MODEL_NAME}")

    texts = chunks_df["chunk_text"].astype(str).tolist()

    embeddings = create_embeddings(texts)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    chunks_df.to_csv(METADATA_PATH, index=False, encoding="utf-8")

    print(f"Created FAISS index with {index.ntotal} vectors.")
    print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")
    print(f"Saved chunk metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()