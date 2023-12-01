import chromadb
import json
import os,random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv("nlq to code\.env")

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

persist_directory='db'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
base_path="scraping\md"
dir_lst=os.listdir("scraping\md")
random.shuffle(dir_lst)
total_token=0
for dir in dir_lst:
        all_text=""
        for file in os.listdir(os.path.join(base_path,dir)):
                with open(os.path.join(base_path,dir,file),'r',encoding="utf8") as f:
                        all_text += f.read()
        total_token+=len(all_text)
        texts=text_splitter.split_text(all_text)
        print(len(all_text))
        print(len(texts))
        for t in texts:
                vectordb=Chroma.from_texts([t], embedding=embeddings,persist_directory=persist_directory)
                vectordb.persist()
                vectordb = None
print(dir_lst)
print(total_token)


