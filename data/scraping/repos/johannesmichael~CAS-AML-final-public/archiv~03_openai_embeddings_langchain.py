#first version where I tried to use langchain with ChromaDB
#but because of dependency issues and lanhchain not keeping up with the latest version of chromaDB
#I decided to use chromaDB directly


import os
import glob
from typing import List
from dotenv import load_dotenv
import openai
from pandas import DataFrame, to_datetime, read_parquet
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
from scripts.constants import CHROMA_SETTINGS

load_dotenv()


OUTLOOK_CONTENT_CONNECTION_STRING = os.environ.get('OUTLOOK_CONTENT_CONNECTION_STRING')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
persist_directory = os.environ.get('PERSIST_DIRECTORY')
chunk_size = 500
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

    #testing
    #df = df[:10]
    

    #create new column with list of dictionaries
    df['content_processed_list'] = df['content_processed'].apply(create_list)
    #drop rows where content_processed_list is empty
    df = df[df['content_processed_list'].map(len) > 0]
    print(f"After removing wrong json content: {str(df.shape)}")
    
    #create new column with text from list of dictionaries                                                                 
    df['text'] = df['content_processed_list'].apply(create_text)

    #prepare for embedding by remmoving unnecessary columns
    df_load = df[['subject', 'content','conversation_id', 'web_link',  'text']]

    return df_load




def get_embedding(content, model="text-embedding-ada-002"):
    text = content
    try:
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except:
        return []
    

def process_data(df):
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

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
    try:
        #Save to Azure Blob Storage
        # Create the BlobServiceClient object which will be used
        blob_service_client = BlobServiceClient.from_connection_string(OUTLOOK_CONTENT_CONNECTION_STRING)

        container_name = 'outlookcontent'
        #get today's date
        today = date.today().strftime('%Y-%m-%d')
        # Create a blob client using the local file name as the name for the blob
        file_name = today + "_outlook_ada_data.parquet"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        

        parquet_file = BytesIO()
        df.to_parquet(parquet_file,  engine='pyarrow')
        parquet_file.seek(0)  # change the stream position back to the beginning after writing

        response = blob_client.upload_blob(data=parquet_file, overwrite=True)

        
    except:
        print("error uploading data to blob storage")

    else:
        return response

def main():
    df = clean_data("2023-07-08final_data.parquet")
    #df = df[:10]
    print(f"DF of shape:  {str(df.shape)}")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        #TODO: adapt to dataframe
        texts = process_data([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_data(df)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run query.py to query your emails")    



if __name__ == "__main__":
    main()