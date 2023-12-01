# pylint: disable=print-used
import argparse
from typing import Final

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import PERSIST_DIRECTORY_BOTO3_DB, PERSIST_DIRECTORY_COMPANY_DB, SOURCE_DIRECTORY_BOTO3_DOCS, \
    SOURCE_DIRECTORY_COMPANY_DOCS
from loaders.boto3_scraper_loader import Boto3ScraperLoader
from loaders.document_loader import DocumentLoader
from schemas.boto3_loader_type import Boto3LoaderType

load_dotenv('.env')

GREEN: Final[str] = '\033[0;32m'


def split_docs(documents, chunk_size=500, chunk_overlap=20):
    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def process_docs_and_persist(loader, persist_directory):
    documents = loader.load()
    print(f'Number of loaded documents are [{len(documents)}] will be ingested to the vector store [{persist_directory}]')

    chunks = split_docs(documents)
    print(f'Number of chunks are [{len(chunks)}]')

    vectordb = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    vectordb.persist()


BOTO3_LOADER_MAP = {
    Boto3LoaderType.DOCUMENTS_LOADER: lambda args: DocumentLoader(documents_path=SOURCE_DIRECTORY_BOTO3_DOCS),
    Boto3LoaderType.SCRAPING_LOADER: lambda args: Boto3ScraperLoader(args.boto3_service_name)
}


def create_boto3_loader(args):
    boto3_loader_type = Boto3LoaderType(args.boto3_loader)
    if boto3_loader_type in BOTO3_LOADER_MAP:
        return BOTO3_LOADER_MAP[boto3_loader_type](args)
    raise ValueError(f'Unsupported loader type: {boto3_loader_type}')


def setup_arg_parser():
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description='Process documents based on specified loader.')

    parser.add_argument('--process-boto3-docs', action='store_true', help='Flag to determine if Boto3 documents should be processed.')

    parser.add_argument('--boto3-loader', choices=[loader.value for loader in Boto3LoaderType],
                        default=Boto3LoaderType.SCRAPING_LOADER.value, help='Specify which loader to use for Boto3 documents.')

    parser.add_argument('--boto3-service-name', type=str, help='Specify the name of the Boto3 service to scrape.')

    return parser.parse_args()


def main():
    args = setup_arg_parser()

    # Process boto3 docs if the flag is set
    if args.process_boto3_docs:
        print(f'{GREEN}Processing Boto3 documents from [{args.boto3_loader}] and persisting to [{PERSIST_DIRECTORY_BOTO3_DB}]')
        process_docs_and_persist(
            loader=create_boto3_loader(args),
            persist_directory=PERSIST_DIRECTORY_BOTO3_DB,
        )

    # Process company docs
    print(f'Processing documents from [{SOURCE_DIRECTORY_COMPANY_DOCS}] and persisting to [{PERSIST_DIRECTORY_COMPANY_DB}]')
    process_docs_and_persist(
        loader=DocumentLoader(documents_path=SOURCE_DIRECTORY_COMPANY_DOCS),
        persist_directory=PERSIST_DIRECTORY_COMPANY_DB,
    )


if __name__ == '__main__':
    main()
