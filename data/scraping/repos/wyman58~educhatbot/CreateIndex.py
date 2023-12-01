from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

data = pd.read_csv('smartchoiceqanda_smartchoicesqanda_captured-list_2023-10-28_7e0cd1d9-1b06-493e-aec4-6351e152e945.csv')

#print(data.head())

docs = []
for index, row in data.iterrows():
    doc = "[Question]:" + row['Question'] + "," + "[Answer]:" + row['Answer'] + "\n\n"
    docs.append(doc)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
)

docs = text_splitter.create_documents(docs)

db = FAISS.from_documents(docs, OpenAIEmbeddings())

db.save_local("QandAIndex")


# query = "What is the smart choice?"

# docs = db.similarity_search(query)

# print(docs)