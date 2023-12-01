# -*- coding: utf-8 -*-

from dotenv import load_dotenv

# Load the API key from an environment variable or file
load_dotenv()


from langchain.document_loaders import PyPDFLoader
url = "https://addi.ehu.es/bitstream/handle/10810/50524/TFG_OihaneAlbizuriSilguero.pdf"
loader = PyPDFLoader(url)
documents = loader.load()
print("Document loaded. Start running the chain...")

print("Splitting and embedding...")
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
# Split the documents into chunks
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

 # Create embeddings and vectorstore 
embeddings = OpenAIEmbeddings()
from langchain.vectorstores import FAISS
docsearch = FAISS.from_documents (texts, embeddings)

# Create LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)


# Create QA chain
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 # search_type="similarity", search_kwargs={"k":2}
                                 retriever=docsearch.as_retriever(), 
                                 chain_type="stuff")

print("Retrieving...")
query = "Brevemente, ¿de qué trata el proyecto? ¿quién lo ha realizado? ¿cuál es su objetivo?"
answer = qa.run(query)
print(answer)


'''
# Run the agent
query = "¿Cuáles son los riesgos del proyecto?"
answer = qa.run(query)
print(answer)
'''

