#!/usr/bin/env python3

# "易速鲜花" 内部员工知识库问答系统
# 1. 加载导入知识库文件
# 2. 将文本分割成多个200字符的片段
# 3. 片段以”嵌入“（Embedding）的形式存入向量数据库（Vector DB）
# 4. 从数据库中检索嵌入片（余弦相似度查找，找到与输入问题类似的嵌入片）
# 5. 将检索到的相似嵌入片与问题一起传给LLM，以获得答案

import os
from dotenv import load_dotenv
load_dotenv()

# 1. Load 导入Document Loaders
from langchain.document_loaders import PyPDFLoader # pip3 install PyPDF
from langchain.document_loaders import Docx2txtLoader # pip3 install Docx2txt
from langchain.document_loaders import TextLoader

# 加载Documents
# base_dir = './static/one-flower' # 文档存放目录
base_dir = './static/one-flower-short-info' # 避免触发OpenAI text-embedding-ada-002模板的rate limit
documents = []
for file in os.listdir(base_dir):
	file_path = os.path.join(base_dir, file)
	if file.endswith('.pdf'):
		loader = PyPDFLoader(file_path)
		documents.extend(loader.load())
	elif file.endswith('.docx') or file.endswith('.doc'):
		loader = Docx2txtLoader(file_path)
		documents.extend(loader.load())
	elif file.endswith('.txt'):
		loader = TextLoader(file_path)
		documents.extend(loader.load())

# 2. Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 切成了一个个200字符左右的文档快
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# pip3 install qdrant-client # Qdrant向量数据库来存储嵌入
# pip3 install tiktoken
# 3. Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Qdrant.from_documents(
	documents = chunked_documents,
	embedding = OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
  location = ":memory:", # in-memory 存储
  collection_name = "my_documents") # 指定collection_name

# Rate limit reached for text-embedding-ada-002 in organization xxx on requests per min. Limit: 3 / min
# 4. Retrieval 准备模型和Retrieval链
import logging
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA # Retrieval链

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever = vectorstore.as_retriever(), llm=llm)

qa_chain = RetrievalQA.from_chain_type(llm, retriever = retriever_from_llm)


# pip3 install flask
# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods = ['GET', 'POST'])
def home():
	if request.method == 'POST':
		question = request.form.get('question')
		result = qa_chain({ "query": question })
		return render_template('index.html', result=result)

	return render_template('index.html')

	if __name__ == "__main__":
		app.run(host = '0.0.0.0', debug = True, port = 5000)

# export FLASK_APP=02-one-flower.py
# flask run
# 本地打开链接：http://127.0.0.1:5000

# web 输入问题: 介绍下网站
# 日志中出现：langchain.retrievers.multi_query:Generated queries: ['1. 可以给我一些关于该网站的详细介绍吗？', '2. 请简要介绍一下这个网站的特点和功能。', '3. 能否提供一些关于该网站的背景信息和运营情况的介绍？']
# web 返回答案: 易速鲜花网站是一个服务型的网站，它提供一个平台，让用户可以在上面购买鲜花。该网站主要提供鲜花的销售服务，用户可以浏览不同种类的鲜花，并选择适合自己需求的花束或花篮。网站上展示了各种不同的花卉，包括玫瑰和康乃馨等。每种花卉都有其特定的象征意义，比如玫瑰代表爱情、激情和美丽，康乃馨代表母爱、纯真和尊重。用户可以根据自己的需求和喜好选择合适的花束，并通过网站进行购买。易速鲜花网站致力于为用户提供高质量的鲜花和优质的服务，让用户能够方便地表达情感和送出美丽的礼物。


