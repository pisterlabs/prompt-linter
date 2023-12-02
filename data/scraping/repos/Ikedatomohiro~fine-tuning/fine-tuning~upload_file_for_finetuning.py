from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY'),
)

response = client.files.create(
  file=open("test_data.jsonl", "rb"),
  purpose="fine-tune"
)

print(response)
