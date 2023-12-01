from dotenv import load_dotenv
load_dotenv()
import os
import openai
import pinecone

openai.api_key = os.environ['OPENAI_API_KEY']

#initialize our pinecone index
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_API_ENV']  # next to api key in console
)
index_name = "iddoc-docs"   

hasIndex = pinecone.list_indexes()
if not hasIndex:
    pinecone.create_index(index_name, dimension=1536, metric="cosine", pod_type="p1")
elif hasIndex[0] != index_name:
    pinecone.delete_index(hasIndex[0])
    #we use dimension 1536 because thats what openai uses
    pinecone.create_index(index_name, dimension=1536, metric="cosine", pod_type="p1")

index = pinecone.Index(index_name)
hasValues = index.describe_index_stats()