"""Paperbot - A tool for searching and summarizing academic papers.

This module provides a command-line interface to build a vector store from a
collection of academic papers in PDF format, and query the store to find relevant
papers and generate summaries using an OpenAI language model.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Union

import textract
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def set_log_level(loglevel: str) -> None:
    """Set the log level for the script.

    Args:
        loglevel (str): Log level as a string (e.g., "WARNING", "INFO", "DEBUG").
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    logging.basicConfig(level=numeric_level)


def check_openai_api_key() -> None:
    """Check if the OpenAI API key is set in the environment variables."""
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OpenAI API key not found. Set the OPENAI_API_KEY environment variable to use OpenAI services.")
        sys.exit(1)


def build_vector_store(
    dirpath: str,
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
    index_dir: str,
    glob: str = "**/*.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 500
) -> Tuple[Chroma, List[str]]:
    """Build a langchain vector store.

    Args:
        dirpath (str): Path to the directory containing PDF files.
        embeddings (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): Embeddings to use for the index.
        index_dir (str): Directory to save the index.
        glob (str, optional): Glob pattern for selecting PDF files. Defaults to "**/*.pdf".
        chunk_size (int, optional): Chunk size for the index. Defaults to 1000.
        chunk_overlap (int, optional): Chunk overlap for the index. Defaults to 500.

    Returns:
        Tuple[Chroma, List[str]]: The built index and list of chunks.
    """
    t0 = datetime.now()
    logger.info("Building vector store")
    logger.info("Loading documents")
    loader = DirectoryLoader(dirpath, glob=glob, loader_cls=PyPDFLoader)
    docs = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    logger.info("Building index")
    index = Chroma.from_documents(chunks, embeddings, persist_directory=index_dir)
    index.persist()

    index_config = {
        "embeddings": embeddings.model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    with open(os.path.join(index_dir, "config.json"), "w") as f:
        json.dump(index_config, f)

    t1 = datetime.now()
    td = (t1 - t0)
    logger.info(f"Vector store saved to '{index_dir}' in {td}.")

    return (index, chunks)


def load_vector_store(index_dir: str) -> Chroma:
    """Load a langchain vector store.

    Args:
        index_dir (str): Directory containing the saved index.

    Returns:
        Chroma: The loaded index.
    """
    logger.info("Loading vector store")
    with open(os.path.join(index_dir, "config.json"), "r") as f:
        index_config = json.load(f)

    embeddings = HuggingFaceEmbeddings(model_name=index_config["embeddings"])
    index = Chroma(persist_directory=index_dir, embedding_function=embeddings)
    return index


def get_similar_documents(index: Chroma, query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Query a langchain vector store.

    Args:
        index (Chroma): The index to query.
        query (str): The query string.
        k (int, optional): Number of similar documents to retrieve. Defaults to 5.

    Returns:
        List[Tuple[str, float]]: List of similar documents and their scores.
    """
    logger.info("Querying vector store")
    t0 = datetime.now()
    docs = index.similarity_search_with_score(query, k=k)
    t1 = datetime.now()
    td = (t1 - t0)
    logger.info(f"Query completed in {td}.")

    return docs


def format_results_for_prompt(results: List[Tuple[str, float]]) -> str:
    """Format doc page_content and metadata into a string to be used as a prompt.

    Args:
        results (List[Tuple[str, float]]): List of documents and their scores.

    Returns:
        str: Formatted string with document metadata and scores.
    """
    prompt = ""
    for result in results:
        doc, score = result
        prompt += f"Metadata: {doc.metadata}\n"
        prompt += f"Score: {score}\n"
        prompt += f"Page Content: {doc.page_content}\n\n"

    return prompt


def answer_question(
    index: Chroma,
    question: str,
    n_sources: int = 5,
    model: str = "gpt3.5-turbo"
) -> Dict[str, Union[str, List[Tuple[str, float]]]]:
    """Retrieve relevant sources from index and answer question.

    Args:
        index (Chroma): The index to query.
        question (str): The question to ask.
        n_sources (int, optional): Number of sources to retrieve. Defaults to 5.
        model (str, optional): OpenAI model to use for question answering. Defaults to "gpt3.5-turbo".

    Returns:
        Dict[str, Union[str, List[Tuple[str, float]]]]: Answer and list of sources.
    """
    results = get_similar_documents(index, question, k=n_sources)

    context = format_results_for_prompt(results)

    prompt_template = """You are paperbot, an AI literature review chatbot who helps users understand collections of academic papers.
    The user has asked the following question: {question}
    The following sources have been retrieved from the literature review collection. Each is a paragraph from a PDF document in the collection. The metadata for each document provides the source file and the page number.
    Please use the sources to provide an answer to the question. Cite the sources and page numbers inline using APA format. Inline citations should be as close as possible to the claim being made, so that it's always clear which references support which claims and no claims are being made without a supporting reference. Include quotations where they would add value. Below your response add a references paragraph linking inline citations to source files. Please indicate where sources disagree.
    Prefer sources with low scores and where possible provide specific examples of experiments which support your answer. Feel free to give lengthy answers.

    Sources:

    {context}

    User Question: {question}
    Paperbot Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm = ChatOpenAI(model_name=model)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run({"context": context, "question": question})

    return {"answer": response, "sources": results}


def main() -> None:
    """Entry point for the Paperbot CLI."""
    parser = argparse.ArgumentParser(description="Paperbot CLI")
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        help="Log level for the script (default: WARNING). Options: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Create index sub-command
    create_index_parser = subparsers.add_parser("create_index", help="Create an index")
    create_index_parser.add_argument("-d", "--input_dir", required=True, help="Input directory containing PDF files")
    create_index_parser.add_argument("-i", "--index_name", required=True, help="Name for the index")
    create_index_parser.add_argument("-s", "--chunk_size", type=int, default=1000, help="Chunk size for the index")
    create_index_parser.add_argument("-o", "--chunk_overlap", type=int, default=500, help="Chunk overlap for the index")
    create_index_parser.add_argument("-e", "--embeddings", choices=["huggingface", "openai"], default="huggingface", help="Embeddings to use for the index (default: huggingface)")

    # Query model sub-command
    query_model_parser = subparsers.add_parser("query_model", help="Query the model")
    query_model_parser.add_argument("-q", "--question", required=True, help="Question to ask the model")
    query_model_parser.add_argument("-i", "--index_name", required=True, help="Name of the index")
    query_model_parser.add_argument("-n", "--n_sources", type=int, default=5, help="Number of sources to retrieve")
    query_model_parser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="OpenAI model to use for question answering (default: gpt-3.5-turbo)")
    query_model_parser.add_argument("-s", "--return_sources", action="store_true", help="Flag to return the sources retrieved from the index")

    args = parser.parse_args()
    set_log_level(args.loglevel)

    if args.command == "create_index":
        if args.embeddings == "openai":
            embeddings = OpenAIEmbeddings()  # Assuming OpenAIEmbeddings class is implemented
        else:
            embeddings = HuggingFaceEmbeddings()

        build_vector_store(
            dirpath=args.input_dir,
            embeddings=embeddings,
            index_dir=os.path.join("indices", args.index_name),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap)

    elif args.command == "query_model":
        index_dir = os.path.join("indices", args.index_name)
        index = load_vector_store(index_dir)
        result = answer_question(index, args.question, n_sources=args.n_sources, model=args.model)
        print(result["answer"])
        if args.return_sources:
            print("\nSources:")
            print(format_results_for_prompt(result["sources"]))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
