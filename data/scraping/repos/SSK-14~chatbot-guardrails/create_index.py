import os
import pickle
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

'''Add the path to your pdf files'''
for file in os.listdir("knowledge_base"):
    file = open("knowledge_base/" + file, "rb")
    reader = PdfReader(file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

'''Divide the input data into chunks
    This will help in reducing the embedding size as we will se in the code
    as well as reduce the token size for the query,'''
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), disallowed_special=())
vectorstore = FAISS.from_texts(texts, embeddings)


with open("vectorstore/index.pkl", 'wb') as f:
    pickle.dump(vectorstore, f)
