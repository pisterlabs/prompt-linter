from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

loader = TextLoader('data/ai.txt')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=chat, retriever=retriever)

query = "When was AI founded as an academic discipline?"

result = qa.run(query)

print(result)


