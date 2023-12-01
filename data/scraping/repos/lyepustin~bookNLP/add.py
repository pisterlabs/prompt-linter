# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description: 

from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
import qdrant_client
import ebooklib
import logging
import os
import sys
import time

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from qdrant_client.http.models.models import Filter
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance, VectorParams

from chat import get_qdrant_client, get_vector_store


def recreate_qdrant_collection(collection_name, size):

    client = get_qdrant_client()
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logging.info(f"'{collection_name}' collection re-created.")
    except Exception as e:
        logging.error(
            f"on create collection '{collection_name}'. " + str(e).replace('\n', ' '))


def split_list_by_length(input_list, max_length):
    sublists, current_sublist, current_length = [], [], 0

    for item in input_list:
        if current_length + len(item) <= max_length:
            current_sublist.append(item)
            current_length += len(item)
        else:
            sublists.append(current_sublist)
            current_sublist, current_length = [item], len(item)

    return sublists + [current_sublist] if current_sublist else []


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=str(os.getenv("TEXT_SPLITTER_SEPARATOR")),
        chunk_size=int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP")),
        length_function=len
    )
    chunks = text_splitter.split_text(str(text))
    return chunks


def add_some_text():
    recreate_qdrant_collection(
        os.getenv("QDRANT_COLLECTION_NAME"), os.getenv("QDRANT_COLLECTION_SIZE"))

    text = os.getenv("TEXT_SAMPLE")
    text_chunks = get_text_chunks(text)
    vector_store = get_vector_store()
    ids = vector_store.add_texts(text_chunks)

    if len(ids) > 1:
        logging.info(
            f"partial content of book '{os.getenv('BOOK_NAME')}' successfully added " +
            f"to the '{os.getenv('VECTOR_DATABASE')}' vector database.")


def get_ebook_chunks(ebook_path):
    raw_text_list = []
    for i, item in enumerate(epub.read_epub(ebook_path).get_items()):
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            raw_content = item.get_body_content().decode('utf-8')
            soup = BeautifulSoup(raw_content, "html.parser")
            paragraphs = soup.find_all("p")
            for paragraph in paragraphs:
                raw_text_list.append(paragraph.get_text())

    return get_text_chunks(" ".join(raw_text_list))


def recurrent_qdrant_add_texts(text_chunks, max_attempts=5):
    attempt = 1
    while attempt <= max_attempts:
        try:
            vector_store = get_vector_store()
            ids = vector_store.add_texts(text_chunks)
            return ids
        except qdrant_client.http.exceptions.ResponseHandlingException as e:
            logging.warning(
                f"[Qdrant Client] Attempt {attempt} to add_texts: Operation timed out, retrying...")
            attempt += 1
            time.sleep(1)
        except Exception as e:
            logging.error(
                f"[Qdrant Client] Attempt {attempt} to add_texts: Encountered an unexpected error: {e}")
            sys.exit()
    if attempt > max_attempts:
        logging.error(
            f"[Qdrant Client] Operation add_texts failed after {max_attempts} attempts.")
        sys.exit()


def get_ebook_title(ebook_path):
    ebook = epub.read_epub(ebook_path)
    return ebook.get_metadata('DC', 'title')[0][0]


def add_full_book():
    recreate_qdrant_collection(
        os.getenv("QDRANT_COLLECTION_NAME"), os.getenv("QDRANT_COLLECTION_SIZE"))

    path = "docs/sherlock-holmes.epub"
    ebook_name = get_ebook_title(path)
    text_chunks = get_ebook_chunks(path)
    splited_text_chunks = split_list_by_length(
        text_chunks, int(os.getenv("OPENAI_EMBEDDINGS_LIMIT_BYTES")))

    for text_chunks in splited_text_chunks:
        logging.info(
            f"adding content to the vector database of size: '{sum(len(text) for text in text_chunks)}'.")
        ids = recurrent_qdrant_add_texts(text_chunks)
        if len(ids) > 1:
            logging.info(
                f"partial content of book '{ebook_name}' successfully added " +
                f"to the '{os.getenv('VECTOR_DATABASE')}' vector database.")

        sleep_time = int(os.getenv("OPENAI_EMBEDDINGS_LIMIT_SECONDS"))
        logging.warning(
            f"sleeping for: '{sleep_time}' secs.")
        time.sleep(sleep_time)


def test():
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl")
    text = "Hi! It's time for the beach"
    text_embedding = embeddings.embed_query(text)
    print(f"Your embedding is length {len(text_embedding)}")
    print(f"Here's a sample: {text_embedding[:5]}...")


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    options = {
        '--full': add_full_book,
        '--some': add_some_text,
        "--test": test
    }
    flag = sys.argv[1]
    if flag in options:
        options.get(flag)()
    else:
        logging.error(
            f"Invalid flag. Please use: " +
            " ".join([key for key, value in options.items()]))


if __name__ == '__main__':
    main()
