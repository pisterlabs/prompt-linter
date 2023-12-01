from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import langchain
import os
import glob
from dotenv import load_dotenv
import openai
import constants
import time

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Set paths
source_path = './data/src/'
destination_file = './data/ingested.txt'
store_path = './vectorstore/'

# Load documents
print("===Loading documents===")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(source_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs=text_loader_kwargs)
documents = loader.load()

if len(documents) == 0:
    print("No new documents found")
    quit()

# Initialize text splitter
print("===Splitting documents into chunks===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
    length_function = len,
    add_start_index = True,
)

split_documents = text_splitter.split_documents(documents)

print("===Embedding text and creating database===")
embeddings = OpenAIEmbeddings(
    show_progress_bar=True,
    request_timeout=60,
)

new_db = FAISS.from_documents(split_documents, embeddings)

print("===Merging new and old database===")
old_db = FAISS.load_local(store_path, embeddings)
old_db.merge_from(new_db)
old_db.save_local(store_path, "index")

# Record the files that we have added
print("===Recording ingested files===")
with open(destination_file, 'a') as f:
    for document in documents:
        f.write(os.path.basename(document.metadata['source']))
        f.write('\n')
