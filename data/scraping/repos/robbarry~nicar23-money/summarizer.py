"""Summarize long documents using langchain.

This example code loads a language model and uses it, along with other packages,
to extract text from a PDF, split it into chunks, and summarize it using a "map
reduce" summarization chain from the langchain library. It then prints the
summary.

Usage:
    summarizer.py <filename>

Options:
    -h --help        Show this screen.
    -t --temperature Set the model temperature  [default: 0.0]
    -c --chunk-size  
        Set the chunk size  [default: 1500]
    -s --separator <separator>      [default: \n]    

Examples:
    summarizer.py pdf "../data/Stroybusinessconsulting judgment.pdf"


OpenAI API key is required in OPENAI_API_KEY environment variable.
See: https://platform.openai.com/signup

"""
# For details on langchain implementation, see:
# https://langchain.readthedocs.io/en/latest/modules/indexes/chain_examples/summarize.html

import argparse

from PyPDF2 import PdfReader

# we are going to use the summarizer chain from langchain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# in conjunction with the OpenAI language model
from langchain import OpenAI

# get env variables
from dotenv import load_dotenv

load_dotenv()


def get_pdf_text(args):
    reader = PdfReader(args.filename)
    pages = [page.extract_text() for page in reader.pages]
    text_blob = "\n".join(pages)
    return text_blob


def get_text(args):
    with open(args.filename, "r") as f:
        text_blob = f.read()
    return text_blob


def summarize(args):
    if args.filename.lower().endswith(".pdf"):
        text_blob = get_pdf_text(args)
    else:
        text_blob = get_text(args)

    text_splitter = CharacterTextSplitter(
        chunk_size=args.chunk_size,
        separator=args.separator,
    )

    texts = text_splitter.split_text(text_blob)
    docs = [Document(page_content=t) for t in texts]

    llm = OpenAI(temperature=args.temperature)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)

    print(f"Summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize long documents using langchain"
    )
    parser.add_argument("filename", help="Path to the file to summarize")
    parser.add_argument(
        "--temperature",
        "-t",
        help="Set the model temperature (0 is deterministic)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        help="Chunk size for the text splitter",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--separator",
        "-s",
        help="Separator for the text splitter",
        default="\n",
    )
    parser.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)
