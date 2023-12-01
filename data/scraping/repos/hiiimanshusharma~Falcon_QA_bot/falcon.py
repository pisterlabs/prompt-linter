import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Load the text from the file
with open(r"ecommerce_Sample_doc.txt") as f:
    text_file = f.read()

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
chunks = text_splitter.split_text(text_file)

# Initialize the embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vectorStore = FAISS.from_texts(chunks, embeddings)

# Initialize the HuggingFaceHub
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 1.0, "max_length": 10000})

# Initialize the retriever
retriever = vectorStore.as_retriever()

# Initialize the RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Streamlit app
st.title("Chat with LangChain QA")

query = st.text_input("Ask a question:")

if query:
    results = chain.run(query)
    st.write(results)
