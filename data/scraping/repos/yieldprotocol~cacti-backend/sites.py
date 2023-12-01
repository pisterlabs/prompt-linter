from typing import Any, Generator, Iterable, List, Optional
import os
import traceback
import uuid

from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from .weaviate import get_client, NAMESPACE_UUID


INDEX_NAME = 'IndexV1'
INDEX_DESCRIPTION = "Index of web3 document chunks"
TEXT_KEY = 'content'
SOURCE_URL_KEY = 'url'
CHUNK_ID_KEY = 'chunk_id'


text_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def delete_schema() -> None:
    client = get_client()
    client.schema.delete_class(INDEX_NAME)


# recreate schema with:
# python3 -c "from index import sites; sites.create_schema(delete_first=True)"

def create_schema(delete_first: bool = False) -> None:
    client = get_client()
    if delete_first:
        delete_schema()
    client.schema.get()
    schema = {
        "classes": [
            {
                "class": INDEX_NAME,
                "description": INDEX_DESCRIPTION,
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text",
                    }
                },
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the chunk",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": TEXT_KEY,
                    },
                    {
                        "dataType": ["text"],
                        "description": "The source url of the chunk",
                        "name": SOURCE_URL_KEY,
                    },
                    {
                        "dataType": ["int"],
                        "description": "The id of the chunk",
                        "name": CHUNK_ID_KEY,
                    },
                ],
            },
        ]
    }
    client.schema.create(schema)


# run with: python3 -c "from index import sites; sites.backfill()"
def backfill():
    from scrape.scrape import get_body_text, has_scrape_error

    client = get_client()
    for i, scraped_url in enumerate(iter_scraped_urls()):
        print('indexing', i, scraped_url.url)
        output = scraped_url.data
        if has_scrape_error(output):
            continue
        try:
            text = get_body_text(output)
        except:
            # sometimes there are .svg or other files in the db
            continue
        metadata = {SOURCE_URL_KEY: scraped_url.url}
        splitted_docs = text_splitter.create_documents([text], metadatas=[metadata])
        splitted_texts = [d.page_content for d in splitted_docs]
        splitted_metadatas = [{CHUNK_ID_KEY: chunk_id, **d.metadata} for chunk_id, d in enumerate(splitted_docs)]
        _add_texts_with_stable_uuids(client, splitted_texts, splitted_metadatas)


def _get_index_size():
    client = get_client()
    data = client.query.aggregate(INDEX_NAME).with_fields('meta { count }').do()
    print(data)
    return data['data']['Aggregate'][INDEX_NAME][0]['meta']['count']


def _add_texts_with_stable_uuids(client: Any, texts: Iterable[str], metadatas: Optional[List[dict]] = None):
    with client.batch as batch:
        ids = []
        for i, doc in enumerate(texts):
            data_properties = {
                TEXT_KEY: doc,
            }
            if metadatas is not None:
                for key in metadatas[i].keys():
                    data_properties[key] = metadatas[i][key]

            source_url = data_properties[SOURCE_URL_KEY]
            chunk_id = data_properties[CHUNK_ID_KEY]
            doc_uuid = uuid.uuid5(NAMESPACE_UUID, f'doc:{source_url}:{chunk_id}')
            batch.add_data_object(data_properties, INDEX_NAME, doc_uuid)


def iter_scraped_urls() -> Generator[Any, None, None]:
    from scrape.models import ScrapedUrl as ScrapedUrlModel
    for scraped_url in _yield_limit(ScrapedUrlModel.query, ScrapedUrlModel.id):
        yield scraped_url


def _yield_limit(qry: Any, pk_attr: Any, maxrq: int = 100) -> Generator[Any, None, None]:
    """specialized windowed query generator (using LIMIT/OFFSET)

    This recipe is to select through a large number of rows thats too
    large to fetch at once. The technique depends on the primary key
    of the FROM clause being an integer value, and selects items
    using LIMIT."""
    # source: https://github.com/sqlalchemy/sqlalchemy/wiki/RangeQuery-and-WindowedRangeQuery

    firstid = None
    while True:
        q = qry
        if firstid is not None:
            q = qry.filter(pk_attr > firstid)
        rec = None
        for rec in q.order_by(pk_attr).limit(maxrq):
            yield rec
        if rec is None:
            break
        firstid = pk_attr.__get__(rec, pk_attr) if rec else None
