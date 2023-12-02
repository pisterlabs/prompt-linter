from functools import partial
from typing import Callable, Dict, List, Tuple

import faiss
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain.llms import AzureOpenAI

from .faiss_qa import embed_file, query_text_qa, retrieve_faiss_indexes_from_text
from .ocr_parser import extract_text_from_img
from .pdf_parser import parse_pdf


def docsearch_create_indexes_from_files(
    num_dimensions: int,
    uploaded_files: List[bytes | str],
    doc_analysis_client: DocumentAnalysisClient,
    embeddings_model: AzureOpenAI,
    text_splitter: Callable[[str], List[str]],
) -> Tuple[faiss.IndexFlatL2, Dict[int, str]]:
    """
    Create Faiss indexes from uploaded files.

    This function takes in the number of dimensions for the Faiss index,
    a list of uploaded files, a document analysis client, an embeddings model
    and a text splitter function. It initializes a Faiss index.

    The function then loops through the uploaded files and checks if the
    file is a PDF or an image. If it is a PDF, it chunks the file and adds it to the index.
    If it is an image, it extracts text from the image and adds it to the index.
    The function returns the Faiss index and a dictionary containing the indexed document store

    :param num_dimensions: number of dimensions for the Faiss index
    :param uploaded_files: list of uploaded files, either bytes or filepaths
    :param doc_analysis_client: document analysis client to use for OCR
    :param embeddings_model: embeddings model to use for embedding
    :param text_splitter: text splitter to use for chunking
    :return: Faiss index and index_doc_store
    """

    # initialise faiss index
    faiss_index = faiss.IndexFlatL2(num_dimensions)

    # create a partial function to pass to add_files_to_index
    img_extract_fn = partial(
        extract_text_from_img,
        document_analysis_client=doc_analysis_client,
    )

    # initialise index_doc_store; this is a dictionary that stores the index:document text
    index_doc_store = {}
    counter = 0

    # loop through files and add them to index
    for uf in uploaded_files:
        if not uf.endswith((".jpg", ".png", ".jpeg", ".pdf")):
            continue

        print(f"Loading Doc: {uf}")
        embedded_texts, file_texts = embed_file(
            uf,
            embeddings_model,
            text_splitter,
            parse_pdf if uf.endswith(".pdf") else img_extract_fn,
        )
        index_doc_store.update({i + counter: t for i, t in enumerate(file_texts)})
        counter += len(file_texts)
        faiss_index.add(embedded_texts)

    return faiss_index, index_doc_store


def docsearch_query_indexes(
    query_text: str,
    faiss_index: faiss.IndexFlatL2,
    index_doc_store: Dict[int, str],
    embeddings_model: AzureOpenAI,
    llm: AzureOpenAI,
) -> str:
    """
    Embeds query text and retrieves the nearest neighbours from the Faiss index
    as well as the corresponding document text.

    Then performs Q/A on the text to return an answer and writes it to the page.
    :param faiss_index: Faiss index to query
    :param index_doc_store: dictionary containing the index:document text
    :param embeddings_model: embeddings model to use for embedding
    :param llm: language model to use for QA
    :return: answer to query
    """

    faiss_idxs = retrieve_faiss_indexes_from_text(
        query_text,
        faiss_index,
        embeddings_model,
    )

    res = query_text_qa(
        query_text,
        index_doc_store,
        llm,
        faiss_idxs,
    )
    return res
