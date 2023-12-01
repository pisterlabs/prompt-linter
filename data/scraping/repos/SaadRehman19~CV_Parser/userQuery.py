import os
import openai
from Embedding import ResumeParser
# from Embedding_V2 import ResumeParser
resume_parser = ResumeParser()
from dotenv import load_dotenv
# from Embedding import search

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key not found.")

openai.api_key = api_key
text='Gaditek is looking for a Senior FullStack Developer to produce scalable software solutions. This position will work alongside the rest of our Development team to deliver timely updates and features to websites, internal tools, and web applications. Must have 3+ years experience of working on PHP/NodeJS/React frameworks. Proficient with relational databases such as MySQL.Proficient parsing and serializing JSON data. Experience using AWS services such as S3, EC2, and RDS.Strong fundamentals on Object Oriented Design skills, Data Structures & Algorithms, Transaction Management.Good communication, presentation, interpersonal, and organizational skills.Comfortable taking calculated risks and embrace fast-paced iterative software developments.'
# text="I want to know someone who has done internship at Tafsol and have a experience in Nodejs"

res=openai.Embedding.create(
    input=text,
    engine="text-embedding-ada-002"
)

embed = res.data[0].embedding
resume_parser.search(embed,text)


