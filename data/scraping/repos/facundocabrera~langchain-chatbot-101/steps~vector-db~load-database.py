###
# SOURCE: https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/chroma/Using_Chroma_for_embeddings_search.ipynb
# ALT: https://supabase.com/blog/openai-embeddings-postgres-vector
###

import chromadb
import pandas as pd
import openai
import os

from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction # chroma is already integrate with OpenAI 
from dotenv import load_dotenv 

load_dotenv()  # This will load the variables from .env file

openai.api_key = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"

# this time we prefer json for the next step
input_datapath = Path(__file__) / ".." / ".." / ".." / "data" / "movies_embedding.json"
input_datapath = input_datapath.resolve()

db_path = Path(__file__) / ".." / ".." / ".." / "data" / "chroma_db"
db_path = db_path.resolve()

if not db_path.exists():
    db_path.mkdir()

# load json into a DataFrame
# ideally we define the id from the start 
df = pd.read_json(str(input_datapath)).assign(id=lambda x: x.index)

df.info(show_counts=True)
# print( df[:5] )
# exit()

chroma_client = chromadb.PersistentClient(path=str(db_path))

embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBEDDING_MODEL)

# if the collection doesn't exist, create it
try:
    movies_collection = chroma_client.get_collection(name="movies", embedding_function=embedding_function)
except ValueError:
    movies_collection = chroma_client.create_collection(name='movies', embedding_function=embedding_function)

    # load the database with the embeddings already calculated
    movies_collection.add(
        ids = df.id.astype(str).tolist(), # the db ask for strings instead of int (not sure why)
        embeddings = df.embedding.tolist(), 
    )

def query_collection(collection, query, max_results, dataframe):
    search_results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    
    # print(search_results)

    movies_df = dataframe[dataframe.id.astype(str).isin(search_results['ids'][0])]

    return movies_df

print("START PLAYING WITH THE STORED RESULTS TO SEE HOW THIS SEARCH WORKS\n")

while True:
    query = input("type your search: ")
    search_results = query_collection(collection=movies_collection, query=query, max_results=10, dataframe=df)
    print(search_results)