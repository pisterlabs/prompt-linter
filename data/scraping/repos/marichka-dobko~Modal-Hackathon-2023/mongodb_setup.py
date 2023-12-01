from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cohere

uri = ""
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

dbname = client['Insurance']
chunks_collection = dbname['Chunks']

api_key = ''
co = cohere.Client(api_key)

#Query
query = "What is the maximum number of visit"
query_embedding = co.embed(texts=[query], model='small').embeddings[0]

results = chunks_collection.aggregate([
    {
        '$search': {
            "index": "ChunksSemanticSearch",
            "knnBeta": {
                "vector": query_embedding,
                "k": 4,
                "path": "embedding"}
        }
    }
])


for document in results:
    print(document['context'])


chunks_doc = chunks_collection.find({})

for doc in chunks_doc:
    context = doc['context']
    doc['embedding'] = co.embed(texts=[context], model='small').embeddings[0]
    chunks_collection.replace_one({'_id': doc['_id']}, doc)



