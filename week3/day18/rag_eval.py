"""A small, end-to-end RAG evaluation example.

The script retrieves three chunks from Qdrant, generates an answer with Groq,
and evaluates the result with five complementary metrics.  It is intentionally
kept in one file so that each step is easy to inspect while learning RAG.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
COLLECTION_NAME = "day18_rag_evaluation"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 3


class KnowledgeChunk(TypedDict):
    """The fields stored for one searchable knowledge chunk."""

    id: str
    text: str
    category: str


class GoldenExample(TypedDict):
    """A labelled example used to measure retrieval and answer quality."""

    question: str
    answer: str
    relevant_chunk_ids: list[str]


class RetrievedChunk(TypedDict):
    """A chunk returned by Qdrant together with its similarity score."""

    id: str
    text: str
    score: float


# This labelled set is the ground truth for this small teaching example.
# Relevant chunk IDs make precision and recall measurable without asking an LLM
# to grade retrieval, while the reference answer grades generated answers.
GOLDEN_DATASET: list[GoldenExample] = [
    {
        "question": "How many paid leave days do employees receive?",
        "answer": "Employees receive 24 days of paid leave per calendar year.",
        "relevant_chunk_ids": ["leave-policy"],
    },
    {
        "question": "Which days do employees work from the office?",
        "answer": "Employees work from the office on Tuesday, Wednesday, and Thursday.",
        "relevant_chunk_ids": ["office-schedule"],
    },
    {
        "question": "What is the monthly gym reimbursement limit?",
        "answer": "Employees can claim up to Rs 3000 per month for gym or fitness membership.",
        "relevant_chunk_ids": ["gym-reimbursement"],
    },
    {
        "question": "How much can I claim for home internet each month?",
        "answer": "Employees can claim up to Rs 2000 per month for home internet.",
        "relevant_chunk_ids": ["internet-reimbursement"],
    },
    {
        "question": "What is the notice period?",
        "answer": "Employees have a 90-day notice period when resigning.",
        "relevant_chunk_ids": ["notice-period"],
    },
    {
        "question": "How long is paid parental leave?",
        "answer": "Eligible employees may take 16 weeks of paid parental leave.",
        "relevant_chunk_ids": ["parental-leave"],
    },
]


def load_knowledge(path: Path = KNOWLEDGE_PATH) -> list[KnowledgeChunk]:
    """Load and validate the JSON chunks that form the knowledge base."""
    with path.open("r", encoding="utf-8") as knowledge_file:
        chunks = json.load(knowledge_file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{path} must contain a non-empty JSON list.")

    required_fields = {"id", "text", "category"}
    for chunk in chunks:
        if not isinstance(chunk, dict) or not required_fields.issubset(chunk):
            raise ValueError(
                f"Every knowledge chunk must contain {sorted(required_fields)}."
            )
    return chunks


def create_clients() -> tuple[QdrantClient, Groq]:
    """Create authenticated Qdrant and Groq clients from environment variables."""
    load_dotenv(BASE_DIR / ".env")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    missing = [
        name
        for name, value in {
            "QDRANT_URL": qdrant_url,
            "QDRANT_API_KEY": qdrant_api_key,
            "GROQ_API_KEY": groq_api_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return (
        QdrantClient(url=qdrant_url, api_key=qdrant_api_key),
        Groq(api_key=groq_api_key),
    )


def prepare_collection(
    qdrant: QdrantClient,
    model: SentenceTransformer,
    chunks: list[KnowledgeChunk],
) -> None:
    """Create the Qdrant collection if needed and upsert current knowledge."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

    embeddings = model.encode([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(
            id=index,
            vector=embedding.tolist(),
            payload=dict(chunk),
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve_chunks(
    question: str,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    """Return the top matching chunks and their Qdrant similarity scores."""
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=model.encode(question).tolist(),
        limit=top_k,
        with_payload=True,
    ).points
    retrieved: list[RetrievedChunk] = []
    for result in results:
        payload = result.payload or {}
        chunk_id = payload.get("id")
        text = payload.get("text")
        if isinstance(chunk_id, str) and isinstance(text, str):
            retrieved.append(
                {"id": chunk_id, "text": text, "score": float(result.score)}
            )
    return retrieved


def generate_answer(question: str, chunks: list[RetrievedChunk], groq: Groq) -> str:
    """Generate an answer using only the chunks retrieved for the question."""
    context = "\n".join(f"- {chunk['text']}" for chunk in chunks)
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied context. If the context "
                    "does not contain the answer, say that you do not know."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty answer.")
    return answer.strip()


def precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate the fraction of retrieved chunks that are relevant."""
    if not retrieved_ids:
        return 0.0
    relevant = set(relevant_ids)
    return sum(chunk_id in relevant for chunk_id in retrieved_ids) / len(retrieved_ids)


def recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate the fraction of all relevant chunks that were retrieved."""
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    return sum(chunk_id in retrieved for chunk_id in set(relevant_ids)) / len(
        set(relevant_ids)
    )


def _judge_score(groq: Groq, criterion: str, prompt: str) -> float:
    """Ask Groq for one 0-1 score and safely extract that score."""
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluator. Return only JSON in the "
                    'form {"score": 0.0}. Score from 0 to 1. ' + criterion
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    match = re.search(r'"score"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)', content)
    if not match:
        raise ValueError(f"Could not parse a 0-1 score from Groq response: {content}")
    return max(0.0, min(1.0, float(match.group(1))))


def faithfulness(question: str, answer: str, context: str, groq: Groq) -> float:
    """Score whether every important claim in the answer is supported by context."""
    return _judge_score(
        groq,
        "Give 1 when the answer is fully supported by the context and 0 when it is unsupported.",
        f"Question: {question}\nContext: {context}\nAnswer: {answer}",
    )


def relevancy(question: str, answer: str, groq: Groq) -> float:
    """Score whether the generated answer directly addresses the question."""
    return _judge_score(
        groq,
        "Give 1 when the answer directly and completely addresses the question without irrelevant content.",
        f"Question: {question}\nAnswer: {answer}",
    )


def correctness(question: str, answer: str, reference: str, groq: Groq) -> float:
    """Score factual agreement between the generated answer and golden answer."""
    return _judge_score(
        groq,
        "Give 1 when the answer is factually equivalent to the reference answer.",
        f"Question: {question}\nReference answer: {reference}\nGenerated answer: {answer}",
    )


def normalize_question(question: str) -> str:
    """Normalize user and golden questions for punctuation-insensitive matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", question.casefold())).strip()


def evaluate_question(
    example: GoldenExample,
    retrieved: list[RetrievedChunk],
    answer: str,
    groq: Groq,
) -> dict[str, float]:
    """Calculate all five metrics for one question and generated answer."""
    context = "\n".join(chunk["text"] for chunk in retrieved)
    retrieved_ids = [chunk["id"] for chunk in retrieved]
    scores = {
        "precision": precision(retrieved_ids, example["relevant_chunk_ids"]),
        "recall": recall(retrieved_ids, example["relevant_chunk_ids"]),
        "faithfulness": faithfulness(example["question"], answer, context, groq),
        "relevancy": relevancy(example["question"], answer, groq),
        "correctness": correctness(
            example["question"], answer, example["answer"], groq
        ),
    }
    return scores


def print_report(
    question: str,
    retrieved: list[RetrievedChunk],
    answer: str,
    scores: dict[str, float],
) -> None:
    """Print retrieved chunks, metric scores, and a simple overall verdict."""
    print("\nTop 3 retrieved chunks:")
    for index, chunk in enumerate(retrieved, start=1):
        print(f"{index}. [{chunk['score']:.3f}] {chunk['text']}")
    print(f"\nAnswer: {answer}")
    print("\nRAG evaluation:")
    for metric, score in scores.items():
        print(f"- {metric.capitalize():13}: {score:.2f}")
    average = sum(scores.values()) / len(scores)
    verdict = "GOOD" if average >= 0.75 else "NEEDS IMPROVEMENT"
    print(f"\nOverall score: {average:.2f}")
    print(f"Final verdict: {verdict}")


def main() -> None:
    """Run one interactive question through retrieval, generation, and evaluation."""
    chunks = load_knowledge()
    qdrant, groq = create_clients()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    prepare_collection(qdrant, embedding_model, chunks)

    question = input("Ask a question: ").strip()
    if not question:
        raise ValueError("A non-empty question is required.")
    retrieved = retrieve_chunks(question, qdrant, embedding_model)
    answer = generate_answer(question, retrieved, groq)

    example = next(
        (
            item
            for item in GOLDEN_DATASET
            if normalize_question(item["question"]) == normalize_question(question)
        ),
        None,
    )
    if example is None:
        raise ValueError(
            "Evaluation requires a question present in GOLDEN_DATASET. "
            "Add a labelled example before evaluating a new question."
        )
    scores = evaluate_question(example, retrieved, answer, groq)
    print_report(question, retrieved, answer, scores)


if __name__ == "__main__":
    main()
