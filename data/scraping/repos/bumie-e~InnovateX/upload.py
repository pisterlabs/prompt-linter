import json
import shutil
import os
import requests
from langchain.document_loaders import TextLoader
import time
from langchain.text_splitter import CharacterTextSplitter
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from decouple import config
import openai

licenseCode = config('OCR_LISENCE_CODE')
UserName =  config('OCR_USERNAME')
supabase_url = config('SUPABASE_URL')
supabase_key = config('SUPABASE_SERVICE_KEY')
OPENAI_API_VERSION = config('OPENAI_API_VERSION')

openai.api_version = config('OPENAI_API_VERSION')
openai.api_key = config('OPENAI_API_KEY')

supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(deployment="chaining",
                              openai_api_version=config('OPENAI_API_VERSION'),
                              openai_api_key = config('OPENAI_API_KEY'),
                            openai_api_base="https://lang-chain.openai.azure.com/",
                            openai_api_type="azure",)

# Initialize text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Convert first 5 pages of multipage document into doc and txt
requestUrl = 'http://www.ocrwebservice.com/restservices/processDocument?language=english&pagerange=1-5&outputformat=txt';

# Path to the folder containing the PDFs

def write_to_db(course, supabase, embeddings):

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name=f"quiz{course}documents", query_name=f"quiz{course}match_documents",)
    path = '/tmp/file.txt'
    try:
        loader = TextLoader(path)
        data = loader.load()
        docs = text_splitter.split_documents(data)
        write_to_supabase(vector_store, docs, course)
    except:
        print('Failed to Write to Supabase')
# Load Quiz


def write_to_supabase(vector_store, docs, course):
    retries = 3
    for _ in range(retries):
        try:
            vector_store.from_documents(docs, embeddings, client=supabase, table_name=f"quiz{course}documents".lower(), query_name=f"quiz{course}match_documents".lower())
            break
        except TimeoutError:
            time.sleep(5)  # Wait for 5 seconds before retrying


def upload_file(image_data, course):
    # with open(os.path.join(dirpath, files), 'rb') as image_file:
    #     image_data = image_file.read()

    r = requests.post(requestUrl, data=image_data, auth=(UserName, licenseCode))

    if r.status_code == 401:
            #Please provide valid username and license code
            print("Unauthorized request")

    # Decode Output response
    jobj = json.loads(r.content)

    ocrError = str(jobj["ErrorMessage"])

    if ocrError != '':
            #Error occurs during recognition
            print ("Recognition Error: " + ocrError)

    #Download output file (if outputformat was specified)
    file_response = requests.get(jobj["OutputFileUrl"], stream=True)
    with open(os.path.join('tmp/', "file.txt"), 'wb') as output_file:
            shutil.copyfileobj(file_response.raw, output_file)
    write_to_db(course, supabase, embeddings)
    
