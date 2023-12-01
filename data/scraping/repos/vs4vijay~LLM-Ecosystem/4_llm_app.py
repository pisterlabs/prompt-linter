#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer

from config import config


def get_embeddings(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"[+] Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    # embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embedding_model = "BAAI/bge-base-en"
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    embeddings_model = SentenceTransformerEmbeddings(client=model)

    return embeddings_model


def get_llm_model():
    print(f"[+] Loading LLM model: {config.openai_model_name}")
    llm = AzureChatOpenAI(
        openai_api_type=config.openai_api_type,
        openai_api_version=config.openai_api_version,
        openai_api_base=config.openai_api_base,
        openai_api_key=config.openai_api_key,
        model_name=config.openai_model_name,
        deployment_name=config.openai_deployment_name,
        verbose=True,
    )

    return llm


def get_vector_store(namespace, embeddings):
    print("[+] Loading vector store: Qdrant")

    # Loading Text Documents
    loader = TextLoader("./data/EngSysQueryLang.txt")
    # loader = DirectoryLoader('./data', glob="*.txt", show_progress=True)
    documents = loader.load()

    # Splitting Text Documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Building Vector Store from Text Documents
    vector_store = Qdrant.from_documents(
        texts,
        embeddings,
        location=":memory:",  # For in-memory storage only
        # path="./data/qdrant.db",        # For on-disk storage only
        # url="http://localhost:6333",    # For remote DB
        collection_name=namespace,
        verbose=True,
    )

    return vector_store


def get_chain():
    embeddings = get_embeddings()
    vector_store = get_vector_store("default", embeddings)
    llm = get_llm_model()

    # results = vector_store.similarity_search_with_score("EngSysLang")
    # print(f"[+] Similarity Search Results: {results=}")

    # PromptTemplate(
    #     input_variables=["name"],
    #     template="""
    #         ## Consider following:
    #         ```
    #         {name}
    #         ```
    #     """,
    # )
    # chain = LLMChain(
    #     llm=llm,
    #     prompt=prompt,
    #     verbose=True,
    # )

    retriever = vector_store.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=True
    )

    return chain


def main():
    print("[+] Running an LLM App")
    chain = get_chain()

    
    while True:
        query = input("Enter your query (or press enter to exit): ")

        if not query:
            break
        
        res = chain.run(
            f"""
                {query}
            """
        )
        print(f"Chain Res: {res=}")
        print('---')


if __name__ == "__main__":
    main()
