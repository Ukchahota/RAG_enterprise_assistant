import os

from dotenv import load_dotenv
from groq import Groq

from src.retriever import retrieve


load_dotenv(override=True)

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_context(retrieved_chunks):
    context_parts = []

    for idx, row in retrieved_chunks.reset_index(drop=True).iterrows():
        source_number = idx + 1

        context_parts.append(
            f"""
[SOURCE {source_number}]
Document: {row['document_name']}
Document type: {row['document_type']}
Page: {row['page_number']}
URL: {row['source_url']}

Text:
{row['chunk_text']}
"""
        )

    return "\n".join(context_parts)


def generate_rag_answer(question: str, top_k: int = 5):
    retrieved_chunks = retrieve(question, top_k=top_k)
    context = build_context(retrieved_chunks)

    system_prompt = """
You are an enterprise university knowledge assistant.

Answer the user's question using only the provided sources.
Do not use outside knowledge.
If the answer is not supported by the sources, say:
"I could not find enough information in the provided documents to answer this."

When possible, mention the document name and page number.
Keep the answer clear, concise, and evidence-based.
"""

    user_prompt = f"""
User question:
{question}

Retrieved sources:
{context}
"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    answer = response.choices[0].message.content

    return answer, retrieved_chunks


def main():
    question = input("Enter your question: ")

    answer, sources = generate_rag_answer(question, top_k=5)

    print("\nGenerated Answer:\n")
    print(answer)

    print("\nSources Used:\n")
    for idx, row in sources.reset_index(drop=True).iterrows():
        print(f"[{idx + 1}] {row['document_name']} - Page {row['page_number']}")
        print(f"Score: {row['score']:.4f}")
        print(f"URL: {row['source_url']}")
        print("-" * 80)


if __name__ == "__main__":
    main()