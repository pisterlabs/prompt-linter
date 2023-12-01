import os
import logging

from langchain import OpenAI
from load_dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


load_dotenv()


if __name__ == "__main__":
    pdf_path = "./pdf/ChainofThought.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # print(document)
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    print(len(docs))

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("./vectorstore/chainofthought")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.run("중요 문장 3개")
    print(res)

    # Ensure the directory exists before writing to the file
    directory = "./pdf_search/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open("./pdf_search/result.txt", "w") as f:
        f.write(res)
