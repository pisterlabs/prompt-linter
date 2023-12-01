from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from dotenv import dotenv_values
import os
from langchain.vectorstores import Chroma

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
# 加載文件夾中的所有txt類型的文件
loader = DirectoryLoader('./data/product/', glob='**/*.txt')
# 將數據轉成 document 對象，每個文件會作為一個 document
documents = loader.load()

# 初始化加載器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加載的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 對象
embeddings = OpenAIEmbeddings()

# 持久化數據
docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="./data/vector_store")
docsearch.persist()
# 加載數據
docsearch = Chroma(persist_directory="./data/vector_store", embedding_function=embeddings)

# 創建問答對象
qa = RetrievalQA.from_chain_type(llm=OpenAI(
), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
# 進行問答
result = qa({"query": "簡介威盛的晶片組產品"})
print(result)