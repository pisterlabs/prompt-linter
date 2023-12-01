from typing import Any, Iterable, List, Optional
import os
import traceback

from langchain.docstore.document import Document
from .weaviate import get_client
from utils import get_widget_index_name
from utils.common import WIDGETS


INDEX_NAME = get_widget_index_name()
INDEX_DESCRIPTION = "Index of widgets"
TEXT_KEY = 'content'


def delete_schema() -> None:
    client = get_client()
    client.schema.delete_class(INDEX_NAME)


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
                ],
            },
        ]
    }
    client.schema.create(schema)


# run with: python3 -c "from index import widgets; widgets.backfill()"
def backfill(delete_first=True):
    # TODO: right now we don't have stable document IDs unlike sites.
    # Always drop and recreate first.
    create_schema(delete_first=delete_first)

    from langchain.vectorstores import Weaviate
    documents = WIDGETS.split("---")
    metadatas = [{} for _ in documents]

    client = get_client()
    w = Weaviate(client, INDEX_NAME, TEXT_KEY)
    w.add_texts(documents, metadatas)
