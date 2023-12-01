import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def faiss_index():
    current_directory = os.getcwd()
    data_path = current_directory + "\\final_project\\Learning_Pathway_Index.csv"
    loader = TextLoader(data_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_learning_path_index")

    new_vectorstore = FAISS.load_local("faiss_learning_path_index", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.run("Give me Machine Learning Course with 10 or 20 min duration.")
    print(res)


if __name__ == "__main__":
    faiss_index()
