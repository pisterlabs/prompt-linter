# -*- coding: utf-8 -*-

# This requires Docker containers for Chroma, Redis and Qdrant to be running.
import os, constants

from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def main():
    # Loading documents from a specified directory
    loader = DirectoryLoader(
        "./data/", glob="./*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()
    # Splitting documents into manageable text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(documents)
    print(
        f"RecursiveCharacterTextSplitter from Langchain:\nProcessed {len(documents)} documents split into {len(all_splits)} chunks"
    )
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
    )
    try:
        # !!! You will need this the first time you run this script as the collection needs to be created !!!
        # Setting up QdrantClient and creating a collection for vector storage
        # url = "35.204.26.135"
        url = "localhost"
        port = "6333"
        collection_name = "instruct-embeddings"
        size=768
        # api_key= "alongpasswordbuteasytoremember!"

        from qdrant_client.http.models import Distance, VectorParams
        try:
            # qdrant_client = QdrantClient(url=url, port=6333, api_key=api_key)
            qdrant_client = QdrantClient(url=url, port=port)
            qdrant_client.delete_collection(
                collection_name=collection_name,
            )
            qdrant_client = QdrantClient(url=url, port=port)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Failed to initialize Qdrant client or create collection: {e}")

        # Initializing Qdrant vectorstore with document embeddings
        # url = "http://localhost:6333"
        vectorstore = Qdrant.from_documents(
            collection_name=collection_name,
            embedding=embeddings,
            documents=all_splits,
            url=url,
            # prefer_grpc=True,
            # api_key=api_key
        )
    except Exception as e:
        print(f"Failed to initialize Qdrant vectorstore: {e}")

    # Loading the Language Model with a callback manager
    llm = Ollama(
        model="openhermes2-mistral",
        verbose=True,
        temperature=0.0,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Setting up a QA Chain with a specific prompt
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )

    def process_llm_response(llm_response):
        print("\n\nSources:")
        for source in llm_response["source_documents"]:
            print(source.metadata["source"])
            
    # Asking a question on the Toolformer PDF
    question = "What is the conclusion summary for the Toolformer whitepaper?"
    print(f"Question: {question}")
    process_llm_response(qa_chain({"query": question}))

    # Asking a question based on the other PDF.
    question = "What is the name of the cat?"
    print(f"Question: {question}")
    process_llm_response(qa_chain({"query": question}))

if __name__ == "__main__":
    main()
