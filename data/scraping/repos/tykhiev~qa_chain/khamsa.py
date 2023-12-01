from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from PyPDF2 import PdfReader
import pickle
import chromadb

chroma_client = chromadb.Client()


openai_api_key = os.getenv("OPENAI_API_KEY")

pdf = PdfReader(
    "C:/Users/USER/OneDrive/Documents/ai-chatbot-3/server/test_data/learn Langchain/KHAMSA CORPORATE PROFILE V2.0 (3).pdf")
text = ""
for page in pdf.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20)
pdf_text = text_splitter.split_text(text=text)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# save FAISS local
docsearch = FAISS.load_local("khamsa", embeddings=embeddings)
store_name = "khamsa"
with open(f"{store_name}.pkl", "wb") as f:
    pickle.dump(docsearch, f)
while True:
    query = input("Enter your query: ")
    docs = docsearch.similarity_search(query=query, k=3)
    chain = load_qa_chain(llm=OpenAI(temperature=0.7), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(question=query, input_documents=docs)
    print(response)
