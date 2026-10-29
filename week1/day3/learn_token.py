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
prompt1="hi"
prompt2="Explain arguments against atheism in detail"
prompt3="Write a 1000 words essay on GANs"
prompts=[prompt1,prompt2,prompt3]
for prompt in prompts:
    message={"role":role,"content":prompt}
    messages=[message]
    response=client.chat.completions.create(model=model,messages=messages,max_tokens=5000)
    #agar max token se jyaada ka output hai tho finsish reason lenght aayega otherwise naturally end hone pe stop
    usage=response.usage
    print(f"Prompt: {prompt} -->your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens} total tokens: {usage.total_tokens}  Finish Reason: {response.choices[0].finish_reason}")
# prompt="what are your views on veganism please answer in 2 lines"
# message={"role":role,"content":prompt}
# messages=[message]
# response=client.chat.completions.create(model=model,messages=messages)
# # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print(response.choices[0].message.content)
