import json
from langchain.document_loaders import TextLoader
import openai
import os

# Set local environment variables
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")

# # Load  document
loader = TextLoader('state_of_the_union.txt')

# Create Index
# Requires chromadb 
# Requires tiktoken
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query =  "What did the president say about Justice Breyer"

# Convert dictionary to string
response =  json.dumps(index.query_with_sources(query))

# Convert string to json
data = json.loads(response)

# Print Values
os.system("clear")

for key, value in data.items():
  print(f"{key}: {value}")