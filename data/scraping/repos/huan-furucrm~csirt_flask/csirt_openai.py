# Helper function: get embeddings for a text
from llama_index.embeddings import openai

def get_embeddings(text):
   response = openai.Embedding.create(
       model="text-embedding-ada-002",
       input = text.replace("\n"," ")
   )
   embedding = response['data'][0]['embedding']
   return embedding