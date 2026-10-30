import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api Key not found")
client=Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"

JD="""
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""
RESUME="""
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""

def ask_llm(system_prompt,user_prompt):
    sys_msg={"role":"system","content":system_prompt}
    user_msg={"role":"user","content":user_prompt}
    messages=[sys_msg,user_msg]

    response=client.chat.completions.create(model=model,messages=messages)
    answer=response.choices[0].message.content
    return answer

def step1_resume_extract():
    system_prompt="""
    You are a professional hr assistant. Extract the skills from the candidate resume provided. Only return the skills no other information. Do not invent skills by yourself. """
    user_prompt=f"extract the skills from the following resume:\n{RESUME}"
    return ask_llm(system_prompt,user_prompt)

def step2_jd_extract():
    system_prompt="""
    You are a professional hr assistant. Extract the skills from the job description provided. Only return the skills no other information. Do not invent skills by yourself. """
    user_prompt=f"extract the skills from the following job description:\n{JD}"
    return ask_llm(system_prompt,user_prompt)

def step3_compare_skills(resume_skills,jd_skills):
    system_prompt="""
    You are a professional hr assistant. Compare the skills extracted from the resume and skills required in the job description.Produce a final score on the basis of comparison. The score should be a number between 0 and 100. Also provide a brief explanation of the score."""

    user_prompt=f"Compare the following skills from resume:\n{resume_skills}\nwith the following skills from job description:\n{jd_skills}\nProvide a score and explanation."
    return ask_llm(system_prompt,user_prompt)

resume_skills=step1_resume_extract()
sleep(2)
jd_skills=step2_jd_extract()    
sleep(2)
score=step3_compare_skills(resume_skills,jd_skills)
print("Resume Skills:",resume_skills)
print("Job Description Skills:",jd_skills)
print("Score and Explanation:",score)
