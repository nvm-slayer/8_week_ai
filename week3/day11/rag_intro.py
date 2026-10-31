import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Load environment variables
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY NOT FOUND")
client=Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"
knowledge_base={"name":"Raj Singh","age":23,"city":"New Delhi","profession":"Software Engineer","hobbies":["coding","reading","watching movies"]  }

#retrieval step

def retrieve_info(query):
    # Simple retrieval logic based on the knowledge base
    query = query.lower()
    if "name" in query:
        return knowledge_base["name"]
    elif "age" in query:
        return knowledge_base["name"] + " is " + str(knowledge_base["age"]) + " years old."
    elif "city" in query:
        return knowledge_base["name"] + " lives in " + knowledge_base["city"] + "."
    elif "profession" in query:
        return knowledge_base["name"] + " is a " + knowledge_base["profession"] + "."
    elif "hobbies" in query:
        hobbies = ', '.join(knowledge_base["hobbies"])
        return knowledge_base["name"] + "'s hobbies include " + hobbies + "."
    else:
        return "I don't have information on that."

#very rigid software, the retrieval matches only the exact keywords, let's say if we ask about how old is raj singh instead of asking "what is raj singh's age?", it will not match the keyword "age" and will return "I don't have information on that.also it can give age of other person also 23 if system prompt is not flexible enough.Knowledge base will also grow too much in size and will be difficult to manage. So we need to make the retrieval step more flexible and robust. We can use embeddings to find the most relevant information from the knowledge base based on the query. This way, we can handle variations in the query and retrieve the correct information."

def ask_llm(question):
    context = retrieve_info(question)
    sys_prompt=f"""answer in one line only,for more information, answer only based on this context do not hallucinate{context}, if answer is publicly available information, answer based on that."""
    sys_message={"role":"system","content":sys_prompt}
    message={"role": "user", "content": question}
    messages=[sys_message,message]
    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer

question="what is my hobbies?"

print(ask_llm(question))