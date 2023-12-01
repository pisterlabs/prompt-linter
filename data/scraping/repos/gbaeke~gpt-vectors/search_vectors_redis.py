import numpy as np
from redis.commands.search.query import Query
import redis
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def search_vectors(query_vector, client, top_k=5):
    base_query = "*=>[KNN 5 @embedding $vector AS vector_score]"
    query = Query(base_query).return_fields("url", "vector_score").sort_by("vector_score").dialect(2)    

    try:
        results = client.ft("posts").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None

    return results



# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')

# Connect to the Redis server
conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)

if conn.ping():
    print("Connected to Redis")

# Enter a query
query = input("Enter your query: ")

# Vectorize the query using OpenAI's text-embedding-ada-002 model
print("Vectorizing query...")
embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")
query_vector = embedding["data"][0]["embedding"]

# Convert the vector to a numpy array
query_vector = np.array(query_vector).astype(np.float32).tobytes()

# Perform the similarity search
print("Searching for similar posts...")
results = search_vectors(query_vector, conn)

if results:
    print(f"Found {results.total} results:")
    for i, post in enumerate(results.docs):
        score = 1 - float(post.vector_score)
        print(f"\t{i}. {post.url} (Score: {round(score ,3) })")
else:
    print("No results found")