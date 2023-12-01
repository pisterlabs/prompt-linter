import argparse

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import NotionDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma


def set_args():
    parser = argparse.ArgumentParser(description="For RAG task")
    parser.add_argument("--dataset_path", type=str, default="notion_db")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--n_gpu_layers", type=int, default=10)
    parser.add_argument("--n_batch", type=int, default=512)
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data2/home/wtzhang/llama-RAG-test/models/ggml-model-q4_0.gguf",
    )
    args = parser.parse_args()
    return args


def load_dataset(args):
    # import notion loader and load md file
    # return the md file
    loader = NotionDirectoryLoader(args.dataset_path)
    pages = loader.load()
    md_file = pages[0].page_content
    return md_file


def splitter(data, args):
    # set headers
    headers_to_split_on = [
        ("#", "Title"),
        ("##", "Section"),
        ("###", "Details"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(data)

    # split recursively
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    all_splits = text_splitter.split_documents(md_header_splits)
    return all_splits


def set_vectorstore(documents, args=None):
    # Build vectorstore and keep the metadata
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=HuggingFaceEmbeddings()
    )
    return vectorstore


def set_llm(args):
    # Define self query retriever
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm


def set_retriever(llm, vectorstore, args=None):
    # Loading llm and create metadata infos
    metadata_field_info = [
        AttributeInfo(
            name="Title",
            description="the first header in the document,introduction",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="Section",
            description="the second header in the document,middle part",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="Details",
            description="the third header in the document,about the details",
            type="string or list[string]",
        ),
    ]
    document_content_description = (
        "the content of the document,related with fruit and vegetable"
    )

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )
    return retriever


if __name__ == "__main__":
    args = set_args()
    # Prepare data
    data = load_dataset(args)
    data_trunked = splitter(data, args)
    # Setup required systems
    vectorstore = set_vectorstore(data_trunked, args)
    # Setup llm model
    llm = set_llm(args)
    # Combine all the components
    retriever = set_retriever(llm, vectorstore, args)
    # Run as a pipeline
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    qa_chain.run("Summarize the Detail part 'Spinach' of the document")
