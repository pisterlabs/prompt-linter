import os,sys
import pathlib
import re,tempfile,pickle

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader,CSVLoader,PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

OUTDIR='embeddings'
PROCESS_DOC='parsed_data.csv'
EMBED_MODEL='text-embedding-ada-002'


def get_file_extension(uploaded_file):
    file_extension =  os.path.splitext(uploaded_file)[1].lower()
    #if file_extension not in [".csv", ".pdf"]:
    #    raise ValueError("Unsupported file type. Only CSV and PDF files are allowed.")
    
    return file_extension

text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 2000,
        chunk_overlap  = 50,
        length_function = len,
    )
file_extension = get_file_extension(PROCESS_DOC)

if file_extension == ".csv":
    loader = CSVLoader(file_path=PROCESS_DOC, encoding="utf-8",csv_args={
        'delimiter': ',',})
    data = loader.load()

elif file_extension == ".pdf":
    loader = PyPDFLoader(file_path=PROCESS_DOC)  
    data = loader.load_and_split(text_splitter)

elif file_extension == ".txt":
    loader = TextLoader(file_path=PROCESS_DOC, encoding="utf-8")
    data = loader.load_and_split(text_splitter)
    
# get cost estimate

import tiktoken

enc = tiktoken.encoding_for_model(EMBED_MODEL)

total_word_count = sum(len(doc.page_content.split()) for doc in data)
total_token_count = sum(len(enc.encode(doc.page_content)) for doc in data)

print(f"\nTotal word count: {total_word_count}")
print(f"\nEstimated tokens: {total_token_count}")
print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")

answer = input("Continue? [y/n] ")
if answer.upper() in ["Y", "YES"]:
    pass
elif answer.upper() in ["N", "NO"]:
    sys.exit(0)



print('\ngenerating vector store...')

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# generate the vector store
vectors = FAISS.from_documents(data, embeddings)

# Save the vectors to a pickle file
with open(f"{OUTDIR}/{PROCESS_DOC.replace('../','')}.pkl", "wb") as f:
    pickle.dump(vectors, f)

print('vector store generated.\n')

