from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader = PyPDFLoader("data/the_last_question.pdf")

pages = loader.load_and_split()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(pages, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=chat, retriever=retriever)

query = "What last question was asked?"

result = qa.run(query)

print(result)

