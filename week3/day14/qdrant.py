import os

from groq import Groq
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Qdrant client
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
print("Qdrant client connected successfully.")

groq_client = Groq(api_key=groq_api_key)
collection_name = "knowledge"
embedding_size=384

# Check if the collection already exists
if qdrant_client.collection_exists(collection_name):
    print("Deleting existing collection...")
    qdrant_client.delete_collection(collection_name)


# Create a new collection with the specified parameters
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)

print(f"Collection '{collection_name}' created successfully.")

#load knowledge base

with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]
# ["line 1 ", "line2", "line3"....]
print(f"Loaded {len(documents)} documents")

#time to create embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# Prepare points for insertion

points=[]

for i,embedding in enumerate(embeddings):
    point = PointStruct(
        id=i+1,
        vector=embedding.tolist(),
        payload={"text": documents[i]}
    )
    points.append(point)

#upload points to Qdrant
qdrant_client.upsert(#upsert is upload and insert, if the point already exists it will be updated, if not it will be inserted
    collection_name=collection_name,
    points=points
)
print(f"Inserted {len(points)} points into collection '{collection_name}' successfully.")

def search(query, top_k=3):

    # Convert the question into an embedding
    query_vector = model.encode(query).tolist()

    # Search Qdrant for similar vectors
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results

query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()


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


question = "How many vacation days do I get? how can i enjoy those vacation days?"

results = search(question, top_k=3)

context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)