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

def llm_answer(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model,messages=messages)
    ans=response.choices[0].message.content
    return ans
bad_prompt=""""

You are a support assistant at a mobile/laptop company
You have to classify the issue in of the 3 category-billing,technical,return
your answer should be in one word only, the one word should be the one of the category given to you in constraints
example-for instance if he wants a refund category is return
if issue is unrelated to any of the categories then mention timepass kahin aur kar
This is a user complaint
My gf broke up with me please help me
"""
print(llm_answer(bad_prompt))