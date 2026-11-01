import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

model=SentenceTransformer('all-MiniLM-L6-v2') 

my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY NOT FOUND")
client=Groq(api_key=my_api_key)
groq_model="openai/gpt-oss-120b"

documents = [
"Employees receive 24 days of paid leave per year.",
"Employees work from the office on Tuesday, Wednesday and Thursday. "
"Monday and Friday are optional work-from-home days.",
"Employees receive Rs 3000 per month for gym reimbursement.",
"Employees can claim Rs 2000 per month for home internet.",
"Employees have a 90 day notice period."
]
document_embeddings = model.encode(documents)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query_embedding):
    scores=[]
    for i, doc_embedding in enumerate(document_embeddings):
        score=cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]
def ask_llm(question,context):
    sys_prompt=f"""answer in one line only,for more information, answer only based on this context do not hallucinate{context}, if answer is publicly available information, answer based on that."""
    sys_message={"role":"system","content":sys_prompt}
    message={"role": "user", "content": question}
    messages=[sys_message,message]
    response=client.chat.completions.create(model=groq_model,messages=messages)
    answer=response.choices[0].message.content
    return answer
query="How much vacation do employees get?"
query_embedding=model.encode(query)

score,context=retrieve(query_embedding)
answer=ask_llm(query,context)
print(answer)
#at its core rag system is like this, but by this method size of embeddings will grow too much also we cant compare line by line and get the highest score, this process is very slow, all these are solved by vector databases like qdrant