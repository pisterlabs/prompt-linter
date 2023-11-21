from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
import characters
import settings
import openai
import os
import json
import helper_module
from pathlib import Path


def load_document_single(filepath: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(filepath)[1]
    loader_class = settings.DOCUMENTS_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(filepath)
    else:
        raise ValueError(f"Unknown document type: {filepath}")
    return loader.load()[0]


def load_documents_batch(filepaths):
    helper_module.log(f"Loading documents in batch: {filepaths}", 'info')
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_document_single, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return data_list, filepaths


def split_documents(documents):
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


def load_documents(source_folder: str):
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_folder):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in settings.DOCUMENTS_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(settings.INGEST_THREADS, max(len(paths), 1))
    chunk_size = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunk_size):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunk_size)]
            # submit the task
            future = executor.submit(load_documents_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def create_vectordb(source_folder):
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)

    # Load documents and split in chunks
    helper_module.log(f"Loading documents from {source_folder}", 'info')
    documents = load_documents(source_folder)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.DOCUMENT_SPLITTER_CHUNK_SIZE,
                                                   chunk_overlap=settings.DOCUMENT_SPLITTER_CHUNK_OVERLAP)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=settings.PYTHON_SPLITTER_CHUNK_SIZE,
        chunk_overlap=settings.PYTHON_SPLITTER_CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    helper_module.log(f"Loaded {len(documents)} documents from {source_folder}", 'info')
    helper_module.log(f"{len(texts)} chunks of text split", 'info')

    embeddings = OpenAIEmbeddings()

    my_vectordb = Chroma.from_documents(
        collection_name=settings.VECTORDB_COLLECTION,
        documents=texts,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_FOLDER,
    )

    return my_vectordb


def get_vectordb(collection_name=settings.VECTORDB_COLLECTION):
    embeddings = OpenAIEmbeddings()
    my_vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=settings.CHROMA_DB_FOLDER,
        embedding_function=embeddings
    )

    return my_vectordb


def delete_vectordb():
    my_vectordb = get_vectordb(settings.VECTORDB_COLLECTION)
    my_vectordb.delete_collection()
    my_vectordb.persist()
    helper_module.log(f"Vector DB collection deleted: {settings.VECTORDB_COLLECTION}", 'info')


def retrieval_qa_run(system_message, human_input, context_memory, callbacks=None):

    my_vectordb = get_vectordb()
    retriever = my_vectordb.as_retriever(search_kwargs={"k": settings.NUM_SOURCES_TO_RETURN})

    template = system_message + settings.RETRIEVER_TEMPLATE

    qa_prompt = PromptTemplate(input_variables=["history", "context", "question"],
                               template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            temperature=settings.DEFAULT_GPT_QA_HELPER_MODEL_TEMPERATURE,
            model_name=settings.DEFAULT_GPT_QA_HELPER_MODEL,
            streaming=True,
            callbacks=callbacks,
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt,
                           "memory": context_memory},
    )
    helper_module.log("Running QA chain...", 'info')
    response = qa_chain(human_input)
    my_answer, my_docs = response["result"], response["source_documents"]
    helper_module.log(f"Answer: {my_answer}", 'info')
    return my_answer, my_docs


def embed_conversations():
    """ Ingest past conversations as long-term memory into the vector DB."""

    helper_module.log(f"Loading conversations in batch: {settings.CONVERSATION_SAVE_FOLDER}", 'info')

    conversations = []
    for json_file in settings.CONVERSATION_SAVE_FOLDER.glob('*.json'):
        if not str(json_file).endswith(settings.SNAPSHOT_FILENAME):
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                result_str = ""
                for entry in json_data:
                    result_str += f"{entry['role']}: {entry['content']}\n"
                conversations.append(Document(page_content=result_str, metadata = {"source": str(json_file)}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.DOCUMENT_SPLITTER_CHUNK_SIZE,
                                                   chunk_overlap=settings.DOCUMENT_SPLITTER_CHUNK_OVERLAP)

    texts = text_splitter.split_documents(conversations)
    my_vectordb = get_vectordb()
    my_vectordb.add_documents(documents=texts, embeddings=OpenAIEmbeddings())
    helper_module.log(f"{len(conversations)} of conversations found", 'info')
    helper_module.log(f"{len(texts)} chunks of text embedded", 'info')


def display_vectordb_info():
    persistent_client = chromadb.PersistentClient(path=settings.CHROMA_DB_FOLDER)
    collection = persistent_client.get_or_create_collection(settings.VECTORDB_COLLECTION)
    helper_module.log(f"VectorDB Folder: {settings.CONVERSATION_SAVE_FOLDER}", 'info')
    helper_module.log(f"Collection: {settings.VECTORDB_COLLECTION}", 'info')
    helper_module.log(f"Number of items in collection: {collection.count()}", 'info')


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    while True:
        display_vectordb_info()
        user_input = input("\n(C)create DB, (E)mbed conversations, (D)elete collection, (Q)uery - type 'quit' or 'exit' to quit: ")
        if 'c' == user_input.lower().strip():
            create_vectordb(settings.DOCUMENT_FOLDER)
        elif 'e' == user_input.lower().strip():
            embed_conversations()
        elif 'd' == user_input.lower().strip():
            user_input = input("\nAre you sure? Type 'yes' if you are: ")
            if 'yes' == user_input.lower().strip():
                delete_vectordb()
        elif 'q' == user_input.lower().strip():
            while True:
                memory = ConversationBufferMemory(input_key="question",
                                                  memory_key="history")
                query = input("\nQuery: ")
                if 'quit' in query or 'exit' in query:
                    break
                helper_module.log(f"Querying model: {settings.DEFAULT_GPT_QA_HELPER_MODEL}", 'info')
                system_input = characters.CUSTOM_INSTRUCTIONS
                answer, docs = retrieval_qa_run(system_input, query, memory)

                # Print the result
                print("\n\n> Question:")
                print(query)
                print("\n> Answer:")
                print(answer)

                if settings.SHOW_SOURCES:
                    print("----------------------------------SOURCE DOCUMENTS---------------------------")
                    for document in docs:
                        print("\n> " + document.metadata["source"] + ":")
                        print(document.page_content)
                    print("----------------------------------SOURCE DOCUMENTS---------------------------")

        elif 'quit' in user_input or 'exit' in user_input:
            break
        else:
            print("Unknown choice.\n")


