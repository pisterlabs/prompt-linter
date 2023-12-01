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


def get_result(query):
#Query
    #query = "I am 29 years old patient concenr about insurance because I do a lot of sport"
    query_embedding = co.embed(texts=[query], model='small').embeddings[0]

    results = chunks_collection.aggregate([
        {
            '$search': {
                "index": "ChunksSemanticSearch",
                "knnBeta": {
                    "vector": query_embedding,
                    "k": 3,
                    "path": "embedding"}
            }
        }
    ])

    res = []
    for document in results:
        res.append(document)

    return res