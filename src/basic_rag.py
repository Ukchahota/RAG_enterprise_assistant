import os
from dotenv import load_dotenv
from openai import OpenAI

from src.retriever import retrieve


load_dotenv(override=True)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = os.getenv("NVIDIA_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")


def build_context(retrieved_chunks):
    """
    Convert retrieved chunks into a structured context block for the LLM.
    """
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


def extract_answer(completion):
    """
    Safely extract the final answer from NVIDIA/OpenAI-compatible response.
    Some reasoning models may return reasoning_content separately, but for this
    project we only want the final answer.
    """
    message = completion.choices[0].message

    answer = getattr(message, "content", None)

    if answer is None or str(answer).strip() == "":
        raise ValueError(
            "NVIDIA returned no final answer. The model may have returned only "
            "reasoning output. Make sure /no_think is included in the prompt "
            "and that extra_body thinking-token settings are removed."
        )

    return str(answer).strip()


def generate_rag_answer(question: str, top_k: int = 5):
    """
    Basic RAG pipeline using NVIDIA NIM:
    1. Retrieve top-k relevant chunks using FAISS.
    2. Send retrieved chunks to NVIDIA Nemotron.
    3. Generate a grounded answer with source references.
    """
    retrieved_chunks = retrieve(question, top_k=top_k)
    context = build_context(retrieved_chunks)

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY was not found. Check your .env file and add "
            "NVIDIA_API_KEY=your_key_here"
        )

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
    )

    system_message = """
/no_think

You are an enterprise university knowledge assistant.

You must answer using ONLY the retrieved sources provided by the system.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer is not clearly supported by the retrieved sources, say:
  "I could not find enough information in the provided documents to answer this."
- Mention the document name and page number when possible.
- Keep the answer clear, concise, and evidence-based.
- Do not include reasoning, thinking steps, or hidden analysis in the final answer.
"""

    user_message = f"""
/no_think

User question:
{question}

Retrieved sources:
{context}

Answer the question using only the retrieved sources.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=700,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False,
    )

    answer = extract_answer(completion)

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