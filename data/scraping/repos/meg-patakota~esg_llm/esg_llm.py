from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


## load all the reports in pdf format
loader = DirectoryLoader('./data', glob="2023-sustainability-accessible-report.pdf", show_progress=True, use_multithreading=True)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=100
)
documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(documents, embeddings)

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    pipeline_kwargs={"max_length": 100},
)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(), return_source_documents=True)

query = "What are the company's climate objectives?"
response = qa_chain({"query": query})
print(response['result'])