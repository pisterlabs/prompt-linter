# To learn how to combine LLMs with my own data.
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

loader = WebBaseLoader("https://github.com/Saffy127")

pages = loader.load_and_split()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(pages, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=chat, retriever=retriever)

query = "How many Repos dos thie GitHub profile have?"

result = qa.run(query)

print(result)

