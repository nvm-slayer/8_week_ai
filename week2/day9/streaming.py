import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
client=Groq(api_key=my_api_key)
print("GROQ_API_KEY found in environment variables. Proceeding with the code execution.")
model="openai/gpt-oss-120b"
prompt="Give a list of top 10 gaming youtubers of india with explanation of 3 lines"
message={"role":"user","content":prompt}
messages=[message]

stream=client.chat.completions.create(model=model,messages=messages,stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)