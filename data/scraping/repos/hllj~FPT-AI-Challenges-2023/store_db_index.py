import os
import openai
import pickle
from dotenv import load_dotenv
from llama_index import (
    ServiceContext,
    GPTVectorStoreIndex,
    LLMPredictor,
    LangchainEmbedding,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
import pinecone
load_dotenv('.env.default')
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Example of loading data
from llama_index import SimpleDirectoryReader
import glob

input_files = []
for file in glob.glob("data/hubduoc/*.txt"):
    input_files.append(file)

print(input_files)
reader = SimpleDirectoryReader(input_files=input_files)

# # Load the data from the text file
documents = reader.load_data()
import re
regex = r"Triệu chứng:\s*(.*?)\n\s*----------";

for d,b in zip(documents,input_files):    
    info = b.split("/")[-1].split(".")[0].split("-")[-1]
    match = re.search(regex, d.text, re.DOTALL)
    symptoms = re.sub(r"[\n]*", "", match.group(1))
    d.metadata = {"Triệu chứng": symptoms,"Nhóm bệnh nhân": info}

# # Initialize LLM and Embedding models
llm_predictor_chat = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
# CohereEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model = "multilingual-22-12",cohere_api_key=os.environ.get('COHERE_API_KEY'))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chat, embed_model=embeddings)

def build_index():
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('INDEX_ENV')
    )

    index_name = os.getenv('INDEX_NAME')

    if not index_name in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768, metric="cosine")
        pinecone_index = pinecone.Index(index_name)

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        create_index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

        print("Index Created")
        return create_index

    #Query Index in case it already exists
    else:

        vector_store = PineconeVectorStore(pinecone.Index(index_name))
        query_index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

        print("Index Queried")
        return query_index
    
if __name__ == "__main__":
    index = build_index()