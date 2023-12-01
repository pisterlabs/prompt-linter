import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.File.create(
    file=open("./dataset-cleaned/train.jsonl", "rb"),
    purpose="fine-tune",
)

print(response)
