import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# get API key from top-right dropdown on OpenAI website

# res = openai.Engine.list()  # check we have authenticated
# print(res)

MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=MODEL
)

print(res)