import os


from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# use python-dotenv to get API key
from dotenv import load_dotenv
load_dotenv()

os.environ.get("OPENAI_API_KEY")



# load whitepaper (already saved in directory) to langchain
pdf_path = "./paper.pdf" 
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

#print(pages)

# creating embeddings and vectorization
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)

# Using embedded DuckDB with persistence: data will be stored in:...

# query chatbot
#query = "What is Bitcoin?"
#query = "What is proof of work?"

query = "What is the double spent problem?"
result = pdf_qa({"question": query})
print("Answer:")
result["answer"]
print(result["answer"])

