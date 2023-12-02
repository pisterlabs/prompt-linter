from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{os.environ.get('ACTIVELOOP_ORGID')}/{activeloop_dataset_name}"
vecdb = DeepLake(dataset_path=dataset_path, embedding=embeddings)

vecdb.add_documents(docs)

retriever = vecdb.as_retriever()

model = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_llm(model, retriever=retriever)

print(qa_chain.run("When was Michael Jordan born?"))

