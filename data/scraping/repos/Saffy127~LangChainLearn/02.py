from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

loader = CSVLoader("data/MOCK_DATA.csv")

pages = loader.load_and_split()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(pages, embeddings)

retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever)

query = "What airport is Kurt nearest to?"

result = qa_chain.run(query)

print(result)
