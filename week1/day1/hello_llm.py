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
prompt="what are your views on veganism please answer in 2 lines"
message={"role":role,"content":prompt}
messages=[message]
response=client.chat.completions.create(model=model,messages=messages)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(response.choices[0].message.content)
