
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv('.env')

# 读取环境变量
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
loader = DirectoryLoader('/Users/lewiszhang/Desktop/', glob='**/*.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(split_docs, embeddings)
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
result = qa({"query": "TabNet算法是什么？"})
print(result)