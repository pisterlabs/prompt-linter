from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import numpy as np

load_dotenv()

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(url, key)

opeani_key = url = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=opeani_key)

response = client.embeddings.create(
    input="text",
    model="text-embedding-ada-002"
)

embedding = np.array(response.data[0].embedding).reshape(1, -1)

def calculate_similarity(embedding, embeddings_list):
    similarities = cosine_similarity(embedding, embeddings_list)
    return similarities[0]

def convert_embedding(str_embedding):
    return np.fromstring(str_embedding.strip("[]"), sep=',', dtype=float)

data = supabase.table("embedings").select("id, embedding").execute().data
db_embeddings = np.array([row['embedding'] for row in data])

db_embeddings = np.array([convert_embedding(row['embedding']) for row in data])
ids = [row['id'] for row in data]

similarities = calculate_similarity(embedding, db_embeddings)
top_indices = np.argsort(similarities)[::-1][:5]

print("Top 5 similar IDs in descending order of similarity:")
for index in top_indices:
    print(f"ID: {ids[index]}, Similarity: {similarities[index].round(4)}")
  