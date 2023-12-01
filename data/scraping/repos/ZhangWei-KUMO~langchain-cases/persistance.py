from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 创建语言模型
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 创建文本加载器
loader = TextLoader('../20report.txt', encoding='utf8')
documents = loader.load()
# 创建文本分割器
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# 将文本分割成小块
texts = text_splitter.split_documents(documents)
# 创建向量存储，整理流程：1. 将文档分成小块 2. 将小块文档创建Embedding 3. 将向量存储
db = Chroma.from_documents(texts, embeddings,persist_directory='db')
db.persist()
db = None
# 创建索引  
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), 
                                 chain_type="stuff", retriever=retriever)
query = "可以跟我说一说如何实现中华民族伟大复兴吗？"
result = qa.run(query)