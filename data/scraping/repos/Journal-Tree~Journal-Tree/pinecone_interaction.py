# Using native pinecone API
import cohere
import numpy as np
import json
import pinecone


# Open the JSON file
with open('../dataset/student_journal.json') as f:
    # Load the JSON data
    data = json.load(f)

    texts = [item['entry'] for item in data['entries']]

# Access the data
print(texts)

co = cohere.Client(COHERE_API_KEY)

embeds = co.embed(
    texts=texts,
    model="large",
    truncate="LEFT"
).embeddings

print(embeds)
#pinecone.deinit()  # De-initialize any existing Pinecone environments
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1-aws")  # Initialize Pinecone with your API key

index_name = "mood-journal-entries"  # Name of the index to create/use

# Check if the index already exists
if index_name not in pinecone.list_indexes():
    print("Creating index")
    pinecone.create_index(index_name, dimension=len(embeds[0]))

# Instantiate the index
index = pinecone.Index(index_name=index_name)

vectors = []
for i in range(len(embeds)):
    vector = embeds[i]  # Get the embedding vector
    metadata = {"original": texts[i]}  # Optional metadata
    vectors.append((str(i), vector, metadata))

index.upsert(vectors)
print(vectors)
n_clusters = 10  # Number of clusters
query_vectors = np.random.randn(n_clusters, len(embeds[0])).tolist()  # Random query vectors for clustering
result = index.query(queries=query_vectors, top_k=10)
parsed_result = result['results']

print(parsed_result)

