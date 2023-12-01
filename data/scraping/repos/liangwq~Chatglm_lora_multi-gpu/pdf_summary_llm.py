# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

loader = PyPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")

## Other options for loaders 
#loader = UnstructuredPDFLoader("/root/autodl-tmp/quantum_algorithms.pdf")
data = loader.load()
# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=8000, chunk_overlap=800)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# 初始化 openai 的 embeddings 对象
embeddings = SentenceTransformerEmbeddings(model_name ='/root/autodl-tmp/piccolo-large-zh')

# load it into Chroma
docsearch = Chroma.from_documents(texts, embeddings)

query = "这篇文章摘要是什么?"
docs = docsearch.similarity_search(query)

# Here's an example of the first document that was returned
print(docs[0].page_content[:450])

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",openai_api_key='',openai_api_base='')

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",openai_api_key='none',openai_api_base='http://localhost:8000/v1')



# verbose=True will output the prompts being sent to the 
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

output = chain.run(texts)

print(output)
