import time
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain import OpenAI

def process_data():
    print("Getting data...")
    loader = DirectoryLoader("documents", glob="**/*.txt")
    documents = loader.load()
    print("Documents loaded.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("Text splitted.")
    
    texts = text_splitter.split_documents(documents)
    
    print(texts)
    
    return texts

def create_retriever(initialize=False):
    COLLECTION_NAME = "supabase_test"
    
     #Supabase PGVector store
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER"),
        host=os.environ.get("PGVECTOR_HOST"),
        port=int(os.environ.get("PGVECTOR_PORT")),
        database=os.environ.get("PGVECTOR_DATABASE"),
        user=os.environ.get("PGVECTOR_USER"),
        password=os.environ.get("PGVECTOR_PASSWORD"),
    )
    
    print("Getting base embeddings from HuggingFace...")
    embeddings = HuggingFaceEmbeddings()
    print("Embeddings loaded.")
    
    if initialize:
        texts = process_data() 

        print("Creating vector store...")
        vector_store = PGVector.from_documents(
            documents=texts, 
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING
        )
        print("Vector store created.")
    else:
        print("Fetching vector store...")
        vector_store = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
    
    qna_retriever = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qna_retriever

def query(prompt, qna_retriever):
    print(f"Answer: {qna_retriever.run(prompt)}")

if __name__ == '__main__':
    INITIALIZE = False # Change to `True` if Supabase PGVector is already initialized.
    
    load_dotenv()
    
    qna_retriever = create_retriever(initialize=INITIALIZE)

    while True:
        prompt = input("Prompt: ")
        
        if prompt == "":
            break
        
        query(prompt, qna_retriever)
        cont = input("Press 'Enter' to prompt again.\n")
        
        if cont != "":
            break
