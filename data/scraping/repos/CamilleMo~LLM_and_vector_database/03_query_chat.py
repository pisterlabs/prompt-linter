import openai
import pinecone
import replicate

import pandas as pd
import sys
import os
from .load_config import load, system_prompt

openai_token = load("config.yaml")["tokens"]["openai"]
pinecone_token = load("config.yaml")["tokens"]["pinecone"]
pinecone_env = load("config.yaml")["parameters"]["pinecone_env"]
openai.api_key = openai_token
pinecone.init(api_key=pinecone_token, environment=pinecone_env)

if len(sys.argv) < 2:
    prompt = "What is the market share of StarTech ?"
else:
    prompt = sys.argv[1]

print("Prompt: " + prompt)

prompt_embedding = openai.Embedding.create(input = [prompt], model = "text-embedding-ada-002")['data'][0]['embedding']

index = pinecone.Index("startech")
result = index.query(
  vector=prompt_embedding,
  top_k=3,
  include_values=False
)
print(result)
# retrieve relevant documents
doc1 = int(result["matches"][0]["id"])
doc2 = int(result["matches"][1]["id"])
doc3 = int(result["matches"][2]["id"])
print(doc1, doc2, doc3)
df = pd.read_csv("data/documents_processed.csv")
doc_1_content = df.loc[df["index"]==doc1, "document"].values[0]
doc_2_content = df.loc[df["index"]==doc2, "document"].values[0]
doc_3_content = df.loc[df["index"]==doc3, "document"].values[0]
print("=========================================")
print("Document 1:")
print(doc_1_content)
print("=========================================")
print("Document 2:")
print(doc_2_content)
print("=========================================")
print("Document 3:")
print(doc_3_content)

prompt = prompt + "\n" + doc_1_content
prompt = prompt + "\n" + doc_2_content
prompt = prompt + "\n" + doc_3_content
# make the LLM call
os.environ['REPLICATE_API_TOKEN'] = load("config.yaml")["tokens"]["replicate"]

output = replicate.run(
    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"prompt": prompt, "system_prompt": system_prompt}
)
# The replicate/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
# for item in output:
#     # https://replicate.com/replicate/llama-2-70b-chat/versions/2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1/api#output-schema
#     print(item, sep=" ")
print("=========================================")
print("Output:")
print("".join(list(output)))
# remove the token from environment variables
