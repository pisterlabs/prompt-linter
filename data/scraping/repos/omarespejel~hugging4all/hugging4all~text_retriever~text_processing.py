from typing import Dict, List

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from termcolor import colored
from utils import (
    TransformersDocsJSONLLoader,
    get_cohere_api_key,
    get_file_path,
    get_openai_api_key,
    get_query_from_user,
    load_config,
)


def load_documents(file_path: str) -> List[Dict]:
    loader = TransformersDocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    return text_splitter.split_documents(data)


def get_chroma_db(embeddings, documents, path):
    config = load_config()
    if config["recreate_chroma_db"]:
        print("Recreating Chroma DB...")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=path,
        )
    else:
        print("Loading existing Chroma DB...")
        return Chroma(persist_directory=path, embedding_function=embeddings)


def select_embedding_provider(provider: str, model: str):
    if provider.lower() == "openai":
        get_openai_api_key()
        return OpenAIEmbeddings(model=model)
    elif provider.lower() == "cohere":
        get_cohere_api_key()
        return CohereEmbeddings(model=model)
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. Supported providers are 'OpenAI' and 'Cohere'."
        )


def process_query(query: str, vectorstore_chroma: Chroma) -> Dict:
    retriever_chroma = vectorstore_chroma.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever_chroma
    )

    return qa_chain.run(query)


def main():
    config = load_config()

    embeddings = select_embedding_provider(
        config["embeddings_provider"], config["embeddings_model"]
    )

    file_path = get_file_path()
    documents = load_documents(file_path)

    chroma_db_path = config["chroma_db_name"]
    vectorstore_chroma = get_chroma_db(embeddings, documents, chroma_db_path)

    print(colored(f"Loaded {len(documents)} documents.", "green"))

    query = get_query_from_user()

    response = process_query(query, vectorstore_chroma)
    print(response.get("result", "No hay resultado"))


if __name__ == "__main__":
    main()
