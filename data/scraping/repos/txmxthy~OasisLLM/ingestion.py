import os
from typing import List
import click
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utilities.constants import (CHROMA_CFG,
                                 DOCTYPE_LOADERS,
                                 CHROMA_PERSIST_DIR,
                                 INGESTION_DIR)

# Config @TODO, Load config from env
chunk_size = 1000
chunk_overlap = 200
embedding_model = "hkunlp/instructor-xl"


def load_files(input_dir: str) -> List[Document]:
    """
    Load all files from the input directory
    :param input_dir: files for ingestion
    :return: the list of loaded files
    """
    files = []
    num_files = len(os.listdir(input_dir))
    with tqdm(total=num_files, desc='Loading files') as pbar:
        for path in os.listdir(input_dir):
            ext = os.path.splitext(path)[1]
            if ext in DOCTYPE_LOADERS:
                file_path = os.path.join(input_dir, path)
                doctype_loader = DOCTYPE_LOADERS[ext]
                if not doctype_loader:
                    raise ValueError(f"Filetype {ext} not supported")
                files.append(doctype_loader(file_path).load()[0])
            pbar.update(1)
    return files


@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
        ]
    ),
    help="Hardware to use, cuda preferred",
)
def ingest_main(device_type):
    """
    Ingest files from the INGESTION_DIR and create a vectordb
    :param device_type:
    :return:
    """
    # Load
    files = load_files(INGESTION_DIR)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(files)
    print(f"Loaded {len(files)} files as {len(chunks)} chunks.")

    # Embed
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device_type},
    )

    # Chroma DB
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=CHROMA_CFG,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    ingest_main()
