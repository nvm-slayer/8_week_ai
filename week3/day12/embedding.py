import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model=SentenceTransformer('all-MiniLM-L6-v2') #vector size of 384

text="My name is Narendra Modi and i India PM"


t1="Dhoni is the best cricketer in the world"
t2="I love youuuuuuuuuuuu"
print(cosine_similarity(model.encode(t1),model.encode(t2)))
# embedding=model.encode(text)

# print(embedding.shape)
# print(embedding[0:10])
