import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY NOT FOUND")
client=Groq(api_key=my_api_key)
model="openai/gpt-oss-20b"
role="user"
prompt="Suggest me a name for my food brand, answer with the name only "
message_system={
    "role":"system",
    "content":"You are an expert marketing manager"
}
message={"role":role,"content":prompt}
messages=[message_system,message]
response=client.chat.completions.create(model=model,messages=messages,temperature=0)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(response.choices[0].message.content)
