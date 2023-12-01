import os
import argparse

from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter, CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def main(args):
    os.environ["OPENAI_API_KEY"] = args.OPEN_API_KEY

    # Load the PDF file (if the file is a URL, load the PDF file from the URL)
    if args.pdf_path.startswith("http"):
        loader = OnlinePDFLoader(args.pdf_url)
    elif args.pdf_path.endswith(".pdf"):
        loader = PyPDFLoader(args.pdf_url)
    else:
        raise ValueError("Please specify the path of the PDF file to be read.")
    pdf_docs = loader.load()

    # Split by separator and merge by character count
    if args.split_mode == "character":
        # Create a CharacterTextSplitter object
        text_splitter = CharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    # Recursively split until below the chunk size limit
    elif args.split_mode == "recursive_character":
        # Create a RecursiveCharacterTextSplitter object
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    elif args.split_mode == "nltk":
        # Create a NLTKTextSplitter object
        text_splitter = NLTKTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    elif args.split_mode == "tiktoken":
        # Create a CharacterTextSplitter object
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        raise ValueError("Please specify the split mode.")

    # Create OpenAIEmbeddings and FAISS objects. Vectorize the chunks created above and save.
    documents = text_splitter.split_documents(pdf_docs)
    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(documents, embeddings)
    faiss_db.save_local(args.faiss_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPEN_API_KEY", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Please specify the chunk_size for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Please specify the chunk_overlap for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--pdf_path", type=str, default="", help="Please specify the path of the pdf to be read.")
    parser.add_argument("--split_mode", type=str, default="recursive_character", help="Please specify the split mode. (character, recursive_character, nltk, tiktoken)")
    parser.add_argument("--faiss_save_path", type=str, default="faiss_index", help="Please specify the name of the created Faiss object.")
    args = parser.parse_args()

    main(args)
