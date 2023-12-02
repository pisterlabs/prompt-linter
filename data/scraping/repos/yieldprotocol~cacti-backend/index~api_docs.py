from typing import Any, Iterable, List, Optional
import os
import traceback
import pathlib
import yaml

from langchain.docstore.document import Document
from .weaviate import get_client


INDEX_NAME = 'APIDocsV1'
INDEX_DESCRIPTION = "Index of API docs"
API_DESCRIPTION_KEY = 'description'
API_SPEC_KEY = 'spec'


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
                        "description": "The API description",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": API_DESCRIPTION_KEY,
                    },
                    {
                        "dataType": ["text"],
                        "description": "The API spec",
                        "name": API_SPEC_KEY,
                    },
                ],
            },
        ]
    }
    client.schema.create(schema)


def backfill():
    # TODO: right now we don't have stable document IDs unlike sites.
    # Always drop and recreate first.
    create_schema(delete_first=True)

    from langchain.vectorstores import Weaviate

    api_docs_dir = pathlib.Path("./knowledge_base/api_docs")

    documents = []
    metadatas = []

    for file_path in api_docs_dir.rglob("*.yaml"):
        with open(file_path, 'r') as file:
            api_doc = yaml.safe_load(file)
            documents.append(api_doc['description'])
            metadatas.append({
                API_SPEC_KEY: api_doc['spec'],
            })

    client = get_client()
    w = Weaviate(client, INDEX_NAME, API_DESCRIPTION_KEY)
    w.add_texts(documents, metadatas)


def test_query():
    client = get_client()

    query_result = client.query\
        .get(INDEX_NAME, [API_DESCRIPTION_KEY, API_SPEC_KEY])\
        .with_near_text({
            "concepts": ["what is the price of"],
            "certainty": 0.85,
        })\
        .with_limit(1)\
        .do()
    print(query_result)
