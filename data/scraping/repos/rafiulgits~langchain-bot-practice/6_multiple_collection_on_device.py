from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import MultiRetrievalQAChain


from dotenv import load_dotenv

load_dotenv(".env.local")


def setup_knowledge(file_path: str, collection_name: str):
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    Chroma.from_documents(
        documents=splitted_documents,
        embedding=embeddings,
        persist_directory="./db/task6",
        collection_name=collection_name,
    )


def get_knowledge(collection_name: str):
    embeddings = HuggingFaceEmbeddings()
    store = Chroma(
        persist_directory="./db/task6",
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return store


def train():
    setup_knowledge("temp/res/about_me.txt", "about-rafiul")
    setup_knowledge("temp/res/product_catalogue.txt", "about-product-catalogue")
    setup_knowledge(
        "temp/res/state-of-the-union-2023.txt", "about-state-of-the-union-2023"
    )


def run():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.1, "max_length": 256},
    )

    knowledge_about_union_state = get_knowledge("about-state-of-the-union-2023")
    knowledge_about_rafiul = get_knowledge("about-rafiul")
    knowledge_about_product_catalogue = get_knowledge("about-product-catalogue")

    retriever_infos = [
        {
            "name": "state of the union",
            "description": "Good for answering questions about the 2023 State of the Union address",
            "retriever": knowledge_about_union_state.as_retriever(),
            "top_k": 2,
        },
        {
            "name": "product catalogue",
            "description": "Good for answering questions about product list and price",
            "retriever": knowledge_about_product_catalogue.as_retriever(),
            "top_k": 1,
        },
        {
            "name": "personal",
            "description": "Good for answering questions about me or rafiul islam",
            "retriever": knowledge_about_rafiul.as_retriever(),
            "top_k": 2,
        },
    ]

    chain = MultiRetrievalQAChain.from_retrievers(
        llm=llm,
        retriever_infos=retriever_infos,
        verbose=True,
    )

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
