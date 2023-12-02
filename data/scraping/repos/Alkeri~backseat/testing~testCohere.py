import cohere
from pymongo import MongoClient
from dotenv import load_dotenv
import os


load_dotenv()


co = cohere.Client(os.getenv('COHERE_API_KEY'))
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['backseat']
collection = db['embeddings']



text = "Add support for tsv Currently we support CSV and JSON. We should add support for TSV"
response = co.embed(
  texts=[text],
  model='small',
)
#print(response)
print(len(response.embeddings[0]))

githubId = "1"

document = {'type': 'issue', 'text': text, 'githubId': githubId, 'cohereSmallEmbedding': response.embeddings[0]}
update = {"$setOnInsert": document}
filter = {'type': 'issue', 'githubId': githubId}

result = collection.update_one(filter, update, upsert=True)





