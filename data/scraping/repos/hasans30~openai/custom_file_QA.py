# tutorial https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Custom%20Files%20Question%20%26%20Answer.ipynb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
# import magic
# import os
# import nltk
from utils import check_and_exit, get_api_key

check_and_exit("OPENAI_API_KEY")
print("OPENAI_API_KEY found")
openai_api_key=get_api_key()
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


# Get your loader ready
loader = DirectoryLoader('data/', glob='**/*.txt', loader_cls=TextLoader)
# Load up your text into documents
documents = loader.load()
# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split your documents into texts
texts = text_splitter.split_documents(documents)
# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)

# Load up your LLM
llm = OpenAI(openai_api_key=openai_api_key)
# Create your Retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# Run a query
query = "did college taught english?"
result = qa({"query": query})
print(result['result'])
# print(result['source_documents'])