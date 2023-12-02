
import os
import glob
import uuid 
from typing import List
import argparse
from dotenv import load_dotenv
import openai
from pandas import DataFrame, to_datetime, read_parquet
import pyarrow
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from azure.storage.blob import BlobServiceClient
import ast
from io import BytesIO
from datetime import date
from tqdm import tqdm
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
from multiprocessing import Pool
from scripts.constants import CHROMA_SETTINGS

load_dotenv()


OUTLOOK_CONTENT_CONNECTION_STRING = os.environ.get('OUTLOOK_CONTENT_CONNECTION_STRING')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
persist_directory = os.environ.get('PERSIST_DIRECTORY')
#setting the chunk size plays a big role in the quality of the answers
chunk_size = 1000
chunk_overlap = 50

tqdm.pandas()



#get data from azure blob storage
def get_data(file_name):
    try:
        # Create the BlobServiceClient object which will be used
        blob_service_client = BlobServiceClient.from_connection_string(OUTLOOK_CONTENT_CONNECTION_STRING)

        container_name = 'outlookcontent'
        
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        
        #download blob
        blob = blob_client.download_blob()
        #convert blob to dataframe
        df = read_parquet(BytesIO(blob.readall()))
        
                
    except: 
        return "error downloading data from blob storage"

    else:
        return df

#create list of dictionaries from json in content_processed column
def create_list(value):
    try:
        return ast.literal_eval(value["choices"][0]["message"]["content"])
    except:
        return []


#create a text from the content keys in the json
def create_text(value):
    output_string = ''
    for d in value:
        for key, value in d.items():
            output_string += str(key) + ': ' + str(value) + '\n'                
        output_string += '\n'  
    return output_string



#clean data and prepare for embedding
def clean_data(file_name):
    df = get_data(file_name)
    print(f"DF of shape:  {str(df.shape)}")
    #drop rows where finish_reason is length
    df = df[df['finish_reason'] != 'length']
    print(f"DF of shape:  {str(df.shape)}")

    # for testing
    #df = df[:10]
    

    #create new column with list of dictionaries
    df['content_processed_list'] = df['content_processed'].apply(create_list)
    #drop rows where content_processed_list is empty
    df = df[df['content_processed_list'].map(len) > 0]
    print(f"After removing wrong json content: {str(df.shape)}")
    
    #create new column with text from list of dictionaries                                                                 
    df['text'] = df['content_processed_list'].apply(create_text)
    df['display_text'] = df['text']

    #prepare for embedding by remmoving unnecessary columns
    df_load = df[['subject', 'content','conversation_id', 'web_link', 'display_text', 'text']]

    return df_load




def get_embedding(content, model="text-embedding-ada-002"):
    text = content
    try:
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except:
        return []
    

def process_data(df, chunk_size, chunk_overlap):
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def request_embeddings(doc):
    model="text-embedding-ada-002"
    return openai.Embedding.create(input=doc.page_content, model=model)

# Create embeddings
def create_embeddings(texts):
    
    
    # Using multiprocessing with 4 processes
    with Pool(4) as p:
        embeddings = list(tqdm(p.imap(request_embeddings, texts), total=len(texts), desc="Creating embeddings"))
    # Extract metadata
    metadatas = [doc.metadata for doc in texts]

    # Create DataFrame
    df_embeddings = DataFrame(metadatas)
    df_embeddings['embedding'] = embeddings
    #create id column
    df_embeddings['uuid'] = [str(uuid.uuid4()) for _ in range(len(df_embeddings.index))]

    return df_embeddings


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False




#function to upload data to azure blob storage
def upload_data(df):
    
    #Save to Azure Blob Storage
    # Create the BlobServiceClient object which will be used
    blob_service_client = BlobServiceClient.from_connection_string(OUTLOOK_CONTENT_CONNECTION_STRING)
    container_name = 'outlookcontent'
    #get today's date
    today = date.today().strftime('%Y-%m-%d')
    # Create a blob client using the local file name as the name for the blob
    file_name = today + "_outlook_ada_embeddings_cs" + str(chunk_size)
    
        
    try:
        extension = '.parquet'
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name+extension)
        parquet_file = BytesIO()
        df.to_parquet(parquet_file,  engine='pyarrow')
        parquet_file.seek(0)  # change the stream position back to the beginning after writing

        

        
    except:
        print("Error uploading to blob storage")

    else:
        return blob_client.upload_blob(data=parquet_file, overwrite=True)
    

# create the top-level parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="file name to download from blob storage, parquet file")
    parser.add_argument("--collection_name", help="Name of the collection to create/use")
    parser.add_argument("--chunk_size", help="chunk_size", default=1000)
    parser.add_argument("--chunk_overlap", help="chunk_overlap", default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    df = clean_data(args.file_name)
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    collection_name = args.collection_name

    #for testing
    #df = df[:100]

    print(f"DF of shape:  {str(df.shape)}")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
    

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        client = chromadb.Client(CHROMA_SETTINGS)
        collections = client.list_collections()
        if "openai_ada_1000cs" in collections:
            collection = client.get_collection(collection)
            #TODO: adapt to dataframe
            #texts = process_data([metadata['source'] for metadata in collection['metadatas']])
            #print(f"Creating embeddings. May take some minutes...")
            #db.add_documents(texts)
        else:
            collection = client.create_collection(name=collection_name, embedding_function=openai_ef)
            
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
        client = chromadb.Client(CHROMA_SETTINGS)
        collection = client.create_collection(name=collection_name, embedding_function=openai_ef)

    texts = process_data(df, chunk_size, chunk_overlap)
    print(f"Creating embeddings. May take some minutes...")
    df_embeddings = create_embeddings(texts)
    upload_data(df_embeddings)
    #function to add to collection
    def add_to_collection(row):
        collection.add(documents=row['content'],
        embeddings=row['embedding']['data'][0]['embedding'],
         metadatas=row[['subject', 'conversation_id', 'web_link', 'display_text']].to_dict(),
        ids=[str(row['uuid'])])
        return True 
    
    #add to collection
    df_embeddings.progress_apply(add_to_collection, axis=1)
        
    print(f"Ingestion complete! You can now run query.py to query your emails")    



if __name__ == "__main__":
    main()


#python 03_openai_ada_embeddings.py --file_name 2021-06-30_outlook_ada_embeddings_csX.parquet --collection_name openai_ada_1000cs --chunk_size 1000 --chunk_overlap 50