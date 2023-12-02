import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

import faiss
import numpy as np
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


def embed_text(text: str, embeddings_model: OpenAIEmbeddings) -> np.ndarray:
    """Embed text with OpenAI embeddings model.
    :param text: text to embed
    :param embeddings_model: OpenAI embeddings model
    :return: np.ndarray of embeddings
    """
    try:
        return embeddings_model.embed_query(text)
    except:
        raise LookupError(f"Error embedding text: {text}")


def embed_text_with_delay(
    text: str, embeddings_model: OpenAIEmbeddings
) -> np.ndarray:
    """Embed text with a 0.9 second delay to prevent rate limiting.
    :param text: text to embed
    :param embeddings_model: OpenAI embeddings model
    :return: np.ndarray of embeddings
    """
    time.sleep(0.9)
    return embed_text(text, embeddings_model)


def embed_file(
    file: str | bytes,
    embeddings_model: OpenAIEmbeddings,
    text_splitter: RecursiveCharacterTextSplitter,
    parsing_fn: Callable,
) -> Tuple[np.ndarray, list[str]]:
    """
    Embeds the text from a file using the provided embeddings model and parsing
    function to extract the file to a list[Document] type object.

    :param file: The file to be embedded, either as a path or as bytes.
    :param embeddings_model: The OpenAI embeddings model to use for embedding the text.
    :param text_splitter: The text splitter to use for splitting the text into chunks.
    :param parsing_fn: The function to use for parsing the file.
    :return: A tuple containing the embedded texts as a numpy array and the split file texts as a list of strings.
    """

    # create partial function to embed text with model
    embed_func = partial(
        embed_text_with_delay, embeddings_model=embeddings_model
    )

    # parse files and upload to index
    file_data = parsing_fn(file)
    file_texts = text_splitter.split_documents(file_data)
    print(f"Chunked {len(file_texts)} chunks of text")

    # embed the chunked texts
    with ThreadPoolExecutor(max_workers=5) as executor:
        embedded_texts = np.array(
            list(
                tqdm(
                    executor.map(
                        embed_func,
                        [p.page_content for p in file_texts],
                    )
                )
            )
        )

    return embedded_texts, file_texts


def retrieve_faiss_indexes_from_text(
    text: str,
    faiss_index: faiss.IndexFlatL2,
    embeddings_model: OpenAIEmbeddings,
    num_nn: int = 5,
) -> np.ndarray:
    """Embeds text and retrieves the num_nn closest results from faiss index.
    :param: text: text to embed
    :param: faiss_index: faiss index to query
    :param: embeddings_model: OpenAI embeddings model
    :param: num_nn: number of nearest neighbors to return
    :return: np.ndarray of faiss indexes
    """

    # get vector embedding from text query
    query_embedding = np.array(embeddings_model.embed_query(text)).astype(
        "float32"
    )
    # expand dimensions to match the right shape
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # query faiss index; k is num nn to return
    _, I = faiss_index.search(query_embedding, num_nn)

    assert I[0].shape == (num_nn,), "Error in retrieving faiss indexes"

    return I[0]  # retrieve first index


def query_text_qa(
    text: str,
    index_doc_store: dict[int, str],
    llm_model: AzureOpenAI,
    faiss_idxs: faiss.IndexFlatL2,
) -> str:
    """Pull text snippets from index doc store based on closest neighbours returned
    from querying faiss index.
    Then apply llm chain to answer question(text).
    :param text: text to query
    :param index_doc_store: dict of index to document
    :param llm_model: llm model to use
    :param faiss_idxs: faiss index to query
    :return: str of answer"""
    # put documents into a list
    qa_docs = [index_doc_store[i] for i in faiss_idxs if i != -1]

    # apply llm chain to answer question completions_llm or chat_llm is applicable
    qa_chain = load_qa_chain(llm_model, chain_type="stuff")
    res = qa_chain.run(input_documents=qa_docs, question=text)

    return res
