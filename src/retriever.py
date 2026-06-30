from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.index"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = pd.read_csv(METADATA_PATH)
    model = SentenceTransformer(MODEL_NAME)

    return model, index, metadata


def retrieve(query: str, top_k: int = 5) -> pd.DataFrame:
    model, index, metadata = load_retriever()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = metadata.iloc[indices[0]].copy()
    results["score"] = scores[0]

    return results[
        [
            "score",
            "document_name",
            "document_type",
            "page_number",
            "chunk_text",
            "source_url",
        ]
    ]


def main():
    query = input("Enter your question: ")

    results = retrieve(query, top_k=5)

    print("\nTop retrieved chunks:\n")

    for i, row in results.iterrows():
        print("=" * 80)
        print(f"Score: {row['score']:.4f}")
        print(f"Document: {row['document_name']}")
        print(f"Type: {row['document_type']}")
        print(f"Page: {row['page_number']}")
        print(f"Source: {row['source_url']}")
        print("\nChunk:")
        print(row["chunk_text"][:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()