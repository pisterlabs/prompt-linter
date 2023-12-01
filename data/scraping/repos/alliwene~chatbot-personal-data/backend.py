import os
import shutil
from typing import List, Generator, Tuple
from tempfile import _TemporaryFileWrapper
from domdf_python_tools.typing import PathLike

from decouple import config  # type: ignore
import hashlib
from memoization import cached
import gradio as gr  # type: ignore

from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever

from result_info import ResultInfo  # type: ignore


#! TODO: use gradio slider for k

metadata_field_info = [
    AttributeInfo(
        name="page",
        description="The page from the document",
        type="integer",
    ),
]
base_llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))

# files_dict: Dict[str, str] = {}
embeddings: CohereEmbeddings = CohereEmbeddings(cohere_api_key=config("COHERE_API_KEY"))
llm_name: str = "gpt-3.5-turbo"


def hash_file(file: _TemporaryFileWrapper) -> str:
    file_name = file.name
    unique_id = hashlib.sha256(file_name.encode()).hexdigest()

    return unique_id


# @cached(max_size=128, thread_safe=False)
def load_db(
    file: _TemporaryFileWrapper,
    document_content_description: str,
    chain_type: str = "stuff",
    k: int = 4,
):
    file_hash: str = hash_file(file)
    file_name: str = file.name
    persist_directory: PathLike = f"chroma/{file_hash}"

    if os.path.exists(persist_directory):
        db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        print(f"Folder {persist_directory} exists!!!")
    else:
        loader = PyPDFLoader(file_name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persist_directory
        )
        print(f"Folder {persist_directory} created!!!")

    # define retriever
    retriever = SelfQueryRetriever.from_llm(
        base_llm,
        db,
        document_content_description,
        metadata_field_info,
        verbose=True,
        enable_limit=True,
        search_kwargs={"k": k},
    )

    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            openai_api_key=config("OPENAI_API_KEY"), model_name=llm_name, temperature=0
        ),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        get_chat_history=lambda h: h,
    )

    return qa


def add_text(history: List[List[str]], query: str):
    history += [(query, "")]
    return history, gr.update(value="", interactive=False)


# @cached(max_size=128, thread_safe=False)
def get_result(
    file: _TemporaryFileWrapper,
    history: List[List[str]],
    document_content_description: str,
    k: int,
) -> ResultInfo:
    print("Get result called!!!")
    qa = load_db(file, document_content_description, k=k)
    result = qa({"question": history[-1][0], "chat_history": history[:-1]})

    return result


def get_combined_result(
    file: _TemporaryFileWrapper,
    history: List[List[str]],
    document_content_description: str,
    k: int,
):
    result = get_result(file, history, document_content_description, k=k)

    for character in result["answer"]:
        history[-1][1] += character

    return history, fmt_search(result["source_documents"])


# def get_response(
#     file: _TemporaryFileWrapper,
#     history: List[List[str]],
#     document_content_description: str,
# ):
#     print(f"File in get response {file}")
#     result = get_result(file, history, document_content_description)

#     for character in result["answer"]:
#         history[-1][1] += character
#         yield history


# def get_source_document(
#     file: _TemporaryFileWrapper,
#     history: List[List[str]],
#     document_content_description: str,
# ) -> str:
#     print(f"File in source docs {file}")
#     result = get_result(file, history, document_content_description)

#     return fmt_search(result["source_documents"])


def fmt_search(docs: List[Document]) -> str:
    result = []
    for i, d in enumerate(docs):
        result.append(
            f"Document {i+1}:\n\n{d.page_content}\n\nMetadata {i+1}:\n\n{str(d.metadata)}"
        )
    return f"\n{'-' * 100}\n".join(result)
