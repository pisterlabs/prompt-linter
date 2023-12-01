from langchain.llms import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv(".env.local")


def setup_knowledge():
    loader = TextLoader("./temp/res/state-of-the-union-2023.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    Chroma.from_documents(
        documents=splitted_documents,
        embedding=embeddings,
        persist_directory="./db/task5",
        collection_name="about_congres",
    )


def get_knowledge():
    embeddings = HuggingFaceEmbeddings()
    store = Chroma(
        persist_directory="./db/task5",
        embedding_function=embeddings,
        collection_name="about_congres",
    )
    return store


def train():
    setup_knowledge()


def run():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.1, "max_length": 256},
    )
    knowledge = get_knowledge()

    # this similarity will help us to find relative doc/resource when we have multiple resources
    retriever = knowledge.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
    input_text = ""
    print("[BOT]: Hello! Ask me about the doc")
    while True:
        input_text = input("[YOU]: ")
        if input_text.lower() == "quit":
            break
        output_text = chain.run(input_text)
        print("[BOT]:", output_text)


if __name__ == "__main__":
    # train()
    run()
