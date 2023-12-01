import ast
import pinecone
import os
import pandas as pd
import numpy as np
import openai
import json
import time

from pathlib import Path
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

PINECONE_KEY = os.environ.get("PINECONE_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")


def parse_embedding(embedding_str):
    return np.array(json.loads(embedding_str))


# Get user inputs
embedding_location = input("Choose embedding location (Local or PineCone): ")
model_choice = input("Choose a command list (All or Enabled): ")
prompt = input("Enter a prompt to test: ")
df = None
# Check if the chosen model is valid
if model_choice not in ["All", "Enabled"]:
    print("Invalid list choice. Please choose a valid command list.")
    exit()
# Check if the chosen model is valid
if embedding_location not in ["Local", "PineCone"]:
    print("Invalid location choice. Please choose a valid location.")
    exit()
if embedding_location == "Local":
    start_time = time.time()
    if model_choice == "All":
        df = pd.read_csv("data/processed/embedding/commands/ada-all-command-embedding.csv", usecols=['embedding', 'name'])

    if model_choice == "Enabled":
        df = pd.read_csv("data/processed/embedding/commands/ada-enabled-command-embedding.csv", usecols=['embedding', 'name'])

    question_vector = get_embedding(prompt, engine="text-embedding-ada-002")
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(np.array(ast.literal_eval(x)),
                                                                           question_vector))
    similar_rows = df.sort_values(by='similarities', ascending=False).head(3)
    for index, row in similar_rows.iterrows():
        print("Command Name:", row['name'])
        print("Similarity:", row['similarities'])
        print()
    print("Time taken: {} seconds".format(time.time() - start_time))

elif embedding_location == "PineCone":
    start_time = time.time()
    pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)
    command_index = pinecone.Index(index_name='enabled-commands-index')
    # Query an example vector
    question_vector = get_embedding(prompt, engine="text-embedding-ada-002")
    similar_rows = command_index.query(vector=question_vector, top_k=5)
    print (similar_rows)
    print("Time taken: {} seconds".format(time.time() - start_time))

# Print the similar intent names and their similarities

