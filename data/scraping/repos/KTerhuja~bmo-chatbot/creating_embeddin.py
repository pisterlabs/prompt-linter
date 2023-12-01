import os 
from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from tqdm import tqdm
from time import sleep
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory

# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma

file_names = os.listdir("./data/")

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size = 1)

for file in ["bmo_ar2022 (2).pdf"]:
    print(file.split(".")[0])
    loader = PyPDFLoader(f"./data/{file}")
    documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,length_function = len)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(folder_path='FAISS_VS', index_name=f"{file.split('.')[0]}_index")
    print(file.split(".")[0])

# docsearch = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name=f"Basel Capital Adequacy Reporting (BCAR) 2023_index")
# retriever = docsearch.as_retriever()
# bmo_retriver = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='bmo_ar2022_index').as_retriever()
# qa_bmo = RetrievalQA.from_chain_type(llm=llm, retriever=bmo_retriver, verbose=True)
# print(qa_bmo.run("Which reports bank BMO has to send to OSFI for BCAR Credit Risk?"))