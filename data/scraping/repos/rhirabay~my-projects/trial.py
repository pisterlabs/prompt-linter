# ----- 環境変数読み込み -----
import os
from dotenv import load_dotenv
load_dotenv()

# ----- OpenAI準備 -----
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----- swagger.yaml読み込み -----
import yaml
import json

with open('./swagger.yaml') as file:
    swagger_yaml = yaml.safe_load(file)
    # print(swagger_yaml['info']['title'])

with open('./swagger.json', 'w', encoding='UTF-8') as file:
    swagger_json = json.dumps(swagger_yaml, indent=2)
    file.writelines(swagger_json)

# ----- lang chain -----
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# ----- json読み込み -----
loader = TextLoader('./swagger.json')
data = loader.load()
print(data)

# ----- 学習 -----
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(data, embedding=embeddings, persist_directory=".data")
vectorstore.persist()

# 学習させる
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(),
    chain_type='refine',
    return_source_documents=True,
    verbose=True
)

print('ok')

# ----- curlコマンドを生成させてみる -----
query = "please show me curl command to create user with name 'hirabay'."
chat_history = []

result = pdf_qa({"question": query, "chat_history": chat_history})

print(result["answer"].replace("\\n", "\n"))