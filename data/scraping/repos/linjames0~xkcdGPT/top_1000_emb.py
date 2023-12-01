import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

with open("top_1000.txt") as f:
    words = [line.strip() for line in f.readlines()]
top_500 = words[:500]
top_501to1000 = words[500:]

print(os.getenv("OPENAI_API_KEY"))

# check if API key is set
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("Ready")
else:
    print("Variable not found")

f.close()

# embed the top 1000 words
top_500_embeddings = {
    word: openai.Embedding.create(
            input=word,
            engine="text-embedding-ada-002",
        )["data"][0]["embedding"] for word in top_500
    }

top_501to1000_embeddings = {
    word: openai.Embedding.create(
            input=word,
            engine="text-embedding-ada-002",
        )["data"][0]["embedding"] for word in top_501to1000
    }

json_str1 = json.dumps(top_500_embeddings)
json_str2 = json.dumps(top_501to1000_embeddings)

# write to file
with open("top_500_emb.json", "w") as f:
    f.write(json_str1)
f.close()

with open("top_501to1000_emb.json", "w") as f:
    f.write(json_str2)
f.close()
