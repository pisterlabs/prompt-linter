from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import os 

load_dotenv()



# Define the list of database directories
#db_directories = ["./chroma_db_swe1_batch_0", "./chroma_db_swe1_batch_1", "./chroma_db_swe1_batch_2", "./chroma_db_swe1_batch_3", "./chroma_db_swe1_batch_4", "./chroma_db_swe1_batch_5", "./chroma_db_swe1_batch_6", "./chroma_db_swe1_batch_7"]
db_directories = [f"test_chroma_db_batch_{x}" for x in range(16)]
embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

# Define your query
#query = "När anses bolaget bildat?"
query = "vad ska jag göra med fastighetstaxeringsavin?"


# Initialize the list to store matching documents
matching_docs = []


# Perform similarity search on each database
for db_directory in db_directories:
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    matching_docs += db.similarity_search(query)

# Tokens
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
au = "Bearer " + HUGGINGFACEHUB_API_TOKEN
API_URL = "https://api-inference.huggingface.co/models/timpal0l/mdeberta-v3-base-squad2"
headers = {"Authorization": au}

#Get response
def query1(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query1({
	"inputs": {
		"question": query,
		"context": matching_docs[0].page_content
	},
})

print(output)



