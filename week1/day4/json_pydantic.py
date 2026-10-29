import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY NOT FOUND")
client=Groq(api_key=my_api_key)
model="openai/gpt-oss-20b"
text="My name is raj.my email is narendramodiresignmaar@gmail.com.my bestfriend is dhruv.I have a washing machine which is not working. My address is ghaziabad.my phone no is 6969420"
class Ticket(BaseModel):
    name:str
    email:str
    issue:str
    phone_no:int
Schema=Ticket.model_json_schema()
response_format={
    "type":"json_object"
}
system_prompt=f""""Extract the personal information from ticket and return strictly based on this schema{Schema} and give me a json output"""
message_system={
    "role":"system",
    "content":system_prompt
}
role="user"
prompt=f""""This is a customer ticket. Please extract the personal information from this {text}
"""
message={"role":role,"content":prompt}
messages=[message_system,message]
response=client.chat.completions.create(model=model,messages=messages,response_format=response_format)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
answer=response.choices[0].message.content
print(answer)
#isko code padhega kaise
import json
raw_json= answer
data_file=json.loads(raw_json)
ticket=Ticket(**data_file)
print(ticket.issue)
print(ticket.phone_no)