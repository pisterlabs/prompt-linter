from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cohere
from postProcess import get_file_content
from pathlib import Path
import time

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



#Uploading to Mongo
def upload_file(filename):
    contents = get_file_content(filename)

    chunks_collection.insert_many(contents)

    chunks_doc = chunks_collection.find({"Filename": filename})

    for doc in chunks_doc:
        context = doc['Context']
        doc['embedding'] = co.embed(texts=[context], model='small').embeddings[0]
        chunks_collection.replace_one({'_id': doc['_id']}, doc)


folder_path = Path('/Users/angky/Cornell/Hackathon/insurance_plans')

# Iterate through all files in the folder
for file_path in folder_path.iterdir():
    if file_path.is_file():
        foldername = '/Users/angky/Cornell/Hackathon/insurance_plans'
        filename = foldername + '/' +  str(file_path.name)
        print("Processing " + filename)
        upload_file(filename)
        time.sleep(30)



