import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
import json

load_dotenv()

QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

qdrant_client=QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
print("Connected to qdrant successfully")

groq_client=Groq(api_key=GROQ_API_KEY)

#creating collection

COLLECTION_NAME="ROMAN_REIGNS"
EMBEDDING_SIZE=384


if qdrant_client.collection_exists(COLLECTION_NAME):
    print("collection already exist")
else:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE),
    )
    print(f"collection created{COLLECTION_NAME}")
qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding model tayyar hai")

texts=[doc["text"] for doc in documents]
embeddings=model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")
#this is creation of points for vector db
points=[]
for i in range(len(documents)):
    point=PointStruct(
        id=i+1,
        vector=embeddings[i].tolist(),
        payload=documents[i]

    )
    points.append(point)

qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

def search(query,top_k=3):
    query_vector=model.encode(query).tolist()

    results=qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    return results

def search_with_filter(query, query_filter=None, top_k=3):
    query_vector = model.encode(query).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points
    print("search with filter executed successfully")
    return results
    

reimbursement_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="reimbursement")
        )
    ]
)


def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"

results = search_with_filter(question,reimbursement_filter, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)
print(context)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)