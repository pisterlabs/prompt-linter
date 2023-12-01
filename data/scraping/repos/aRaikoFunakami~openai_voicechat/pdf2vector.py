''' 
persist_directory に指定した場所に
files で指定した PDF ファイルの
vectorstore DB を保存する
model_name = "gpt-4-0613"

再利用する場合は下記のようにDBの保存場所をpersist_directoryに指定して呼び出す
例)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_split_documents") 

'''
import os
import sys
import platform
import logging
import json

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

def load_config():
    args = sys.argv
    config_file = os.path.dirname(__file__) + "/config.json" if len(args) <= 1 else args[1]
    logging.info(config_file)
    with open(config_file, 'r') as file:
        config = json.load(file)
    return {
        "openai_api_key": config['openai_api_key'],
    }

# Preprocessing for using Open　AI
config = load_config()
openai.api_key = config["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai.api_key

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
embeddings = OpenAIEmbeddings()

# load pdf file
files = [
	"NX350-NX250_UG_JP_M78364_1_2303.pdf",
	"NX350-NX250_OM_JP_M78364V_1_2303.pdf",
	"NX350-NX250_MM_JP_M78364N_1_2303.pdf",
]
# persist_directory="./chroma_split_documents"
persist_directory="./chroma_load_and_split"

pages = []
for file in files:
	pdf_file = os.path.dirname(__file__) + f"/templates/{file}"
	loader = PyPDFLoader(pdf_file)
	# PyPDFLoaderのsplit機能をそのまま利用する場合
	pages = pages + loader.load_and_split()
	# chunk_size で指定したテキストに分割して利用する
	#documents = loader.load_and_split()
	#text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
	#pages = pages + text_splitter.split_documents(documents)
	print(len(pages))


# Stores information about the split text in a vector store
# 保存していたファイルとpagesの両方から vectorstore を作成する
# vectorstore.persist() で追加した pages のデータを含めてファイルにvector情報が保存される
# 連続で persist を呼び出すと
vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()