"""
RAG using Vector Databases. 
"""

import openai
import os
import re
import random

from datetime import datetime
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

from dotenv import load_dotenv
load_dotenv()

CONTEXT_FILE_PATH = "./data/nlp_with_pytorch.txt"
BREAKDOWN_PROMPT_FILE_PATH = "./prompts/vectordb-breakdown-prompt.txt"
PROMPT_FILE_PATH = "./prompts/vectordb-prompt.txt"
RESPONSE_FILE_PATH = lambda timestamp: f"./responsebuffers/{timestamp} (vectordb).txt"
TOPIC = "Feed Forward Neural Networks in NLP using PyTorch"

prompt = str()
with open(BREAKDOWN_PROMPT_FILE_PATH, "r+") as file:
  prompt = file.read()

prompt = prompt.replace("<<TOPIC>>", TOPIC)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORGANIZATON"))

completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {
      "role": "system",
      "content": f"You are a highly skilled subject matter expert in {TOPIC}.",
    },
    {
      "role": "user", 
      "content": prompt
    },
  ],
  temperature=0,
  max_tokens=500,
)

# Chunk Processing
chunks = list()
raw_chunks = completion.choices[0].message.content.split("\n")
for chunk in raw_chunks:
  sub_chunks = re.split(r'[.?!;]', chunk)
  chunks.extend([sub.strip() for sub in sub_chunks if sub.strip() and not sub.strip().isdigit()])

loader = TextLoader(CONTEXT_FILE_PATH)
pages = loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
qdrant = Qdrant.from_documents(
  docs,
  embeddings,
  path="./tmp/nlp_with_pytorch",
  collection_name="nlp_with_pytorch",
)

context = str()
relevance_docs_with_scores = [qdrant.similarity_search_with_score(chunk) for chunk in random.choices(chunks, k=5)]
for doc in relevance_docs_with_scores:
  context += doc[0][0].page_content
  context += "\n"

# Context Embedded Prompt
prompt = str()
with open(PROMPT_FILE_PATH, "r+") as file:
  prompt = file.read()
  prompt = prompt.replace("<<CONTEXT>>", context)

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": f"You are a highly skilled subject matter expert in {TOPIC} and the best question architect.",
    },
    {
      "role": "user", 
      "content": prompt
    },
  ],
  temperature=0,
  max_tokens=4096,
)

response = completion.choices[0].message.content

with open(RESPONSE_FILE_PATH(datetime.now().strftime("%d-%m-%Y %H:%M:%S")), "w+") as file:
  file.write(response)

