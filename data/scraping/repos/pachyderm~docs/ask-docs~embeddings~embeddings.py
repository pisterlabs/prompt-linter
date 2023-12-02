import os
from dotenv import load_dotenv
import time

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone 
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone 


load_dotenv()
openai_key = os.environ.get('OPENAI_API_KEY')
pinecone_key = os.environ.get('PINECONE_API_KEY')
pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index = "langchain1"

docs_index_path = "./docs.json" 
docs_index_schema = ".[]" # [{"body:..."}] -> .[].body; see JSONLoader docs for more info
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0,)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["relURI"] = record.get("relURI")
    return metadata

loader = JSONLoader(docs_index_path, jq_schema=docs_index_schema, metadata_func=metadata_func, content_key="body") 

data = loader.load()
texts = text_splitter.split_documents(data) 

pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_environment,
)

if pinecone_index in pinecone.list_indexes():
    print(f'The {pinecone_index} index already exists! We need to replace it with a new one.')
    print("Erasing existing index...")
    pinecone.delete_index(pinecone_index) 

time.sleep(60)
print("Recreating index...")
# wait a minute for the index to be deleted
pinecone.create_index(pinecone_index, metric="cosine", dimension=1536, pods=1, pod_type="p1") 


if pinecone_index in pinecone.list_indexes():

    print(f"Loading {len(texts)} texts to index {pinecone_index}... \n This may take a while. Here's a preview of the first text: \n {texts[0].metadata} \n {texts[0].page_content}")

    for chunk in chunks(texts, 25):
        for doc in chunk:
            if doc.page_content.strip(): 
                print(f"Indexing: {doc.metadata['title']}")
                print(f"Content: {doc.page_content}")
                Pinecone.from_texts([doc.page_content], embedding=embeddings, index_name=pinecone_index, metadatas=[doc.metadata])
            else:
                print("Ignoring blank document")
    print("Done!")  
