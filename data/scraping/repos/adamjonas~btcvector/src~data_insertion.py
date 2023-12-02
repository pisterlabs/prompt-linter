import json
import time
import uuid

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm

from src.config import ES_INDEX, OPENAI_API_KEY, OPENAI_EMBED_MODEL
from src.logger import LOGGER


openai.api_key = OPENAI_API_KEY


def extract_metadata(data):
    # TODO:- Add `indexed_at` field
    url = data["url"] if data["url"] else "No url in data"
    id_ = data["es_id"]  # Maintain ElasticSearch ID in pinecone
    domain = data["domain"] if data["domain"] else ""
    created_at = data["created_at"] if data["created_at"] else ""
    metadata_id = data["id"] if data["id"] else ""
    title = data["title"] if data["title"] else ""
    type_ = data["type"] if data["type"] else ""
    authors = data["authors"] if data["authors"] else ""
    meta_data = {
        "domain": domain,
        "created_at": created_at,
        "id": metadata_id,
        "title": title,
        "type": type_,
        "url": url,
        "authors": authors,
    }
    return id_, meta_data


def data_embed_insertion(es, pinecone_index, data):
    LOGGER.info(
        "Generating of embeddings of text and insertion of data started..."
    )
    start_time = time.time()
    completed_row_count = 0
    batch_size = 50
    texts = []
    metadatas = []
    ids = []
    embed_model = OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL, openai_api_key=OPENAI_API_KEY
    )

    def truncate_text(text_, url):
        return (
            text_[:1000]
            + f"...for more details you can click on below link\n {url}"
            if len(text_.encode("utf-8")) >= 30000
            else text_
        )

    def update_query(document_ids, x):
        return {
            "script": {
                "source": f"ctx._source.upload_to_pinecone = {x}",
                "lang": "painless",
            },
            "query": {"terms": {"_id": document_ids}},
        }

    for i in tqdm(range(len(data))):
        LOGGER.info(f"Processing row number: {i}")

        text = data[i]["clean_text"]
        text = [text] if isinstance(text, str) else text
        id_, metadata = extract_metadata(data[i])
        record_metadata = [
            {
                "chunk": j,
                "text": truncate_text(text_, data[i]["url"]),
                **metadata,
            }
            for j, text_ in enumerate(text)
        ]
        ids.extend([id_ + '_' + uuid.uuid4()] * len(text))
        texts.extend(text)
        metadatas.extend(record_metadata)
        if len(texts) >= batch_size:
            embeds = embed_model.embed_documents(texts)
            try:
                pinecone_index.upsert(vectors=zip(ids, embeds, metadatas))
                LOGGER.info(
                    f"{completed_row_count} rows inserted into pinecone."
                )
                es.update_by_query(index=ES_INDEX, body=update_query(ids, 0))
                LOGGER.info(f"all chunks updated till document #{i}")
            except Exception as e:
                LOGGER.error(f"error uploading the documents: {e}")
                return

            ids = []
            texts = []
            metadatas = []

        completed_row_count = i

    if len(texts) > 0:
        embeds = embed_model.embed_documents(texts)
        try:
            pinecone_index.upsert(vectors=zip(ids, embeds, metadatas))
            es.update_by_query(index=ES_INDEX, body=update_query(ids, 0))
            LOGGER.info("all documents updated")
        except Exception as e:
            LOGGER.error(f"error uploading the documents: {e}")
            return

    LOGGER.info(
        f"Embeddings and insertion upto {completed_row_count}"
        f" is completed and it took {time.time() - start_time:.2f} secs"
    )
    return completed_row_count


def pinecone_data_insertion(es, json_data_path, pinecone_index):
    LOGGER.info("Generation and Insertion of embeddings of all data started...")
    start_time = time.time()
    file = open(json_data_path)
    data = json.load(file)
    file.close()
    data_embed_insertion(es, pinecone_index, data)
    LOGGER.info(
        f"Generation and Insertion of embeddings of all data completed and "
        f"it took {time.time() - start_time:.2f} seconds."
    )
