import os

from langchain.llms import OpenAI

from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, create_vectorstore_agent
from langchain.document_loaders import CSVLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-*")


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

def load_and_chat_csv_file():
    persist_directory = "talk_to_csv_db"
    # Check if embeddings already generated
    # if os.path.exists(persist_directory):
    #     vectordb = Chroma(persist_directory=persist_directory)
    # else:
    csv_loaded = CSVLoader('./transactions.tsv', csv_args={'delimiter': '\t'})  #
    docs = csv_loaded.load()
    docs_splitted = split_docs(docs, chunk_size=1000, chunk_overlap=20)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs_splitted,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    vector_store_info = VectorStoreInfo(vectorstore=vectordb, name="transactions_db",
                                        description="Blockchain transactions of Tips wallet address")
    _llm = OpenAI(temperature=0.1, verbose=True)
    toolkit = VectorStoreToolkit(vectorstore_info=vector_store_info)
    agent_executor = create_vectorstore_agent(
        llm=_llm,
        toolkit=toolkit,
        verbose=True
    )

    print('🦜🔗 GPT CSV Analyzer')
    response = agent_executor.run("What is the total amount of transactions?")
    print(response)


if __name__ == "__main__":
    load_and_chat_csv_file()

