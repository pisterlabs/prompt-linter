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
pdf_path = "./curve-stablecoin.pdf" 
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()


# creating embeddings and vectorization
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)

# query
query = "What is Curve Stablecoin?"
result = pdf_qa({"question": query})
print(query)
print("Answer:")
result["answer"]
print(result["answer"])