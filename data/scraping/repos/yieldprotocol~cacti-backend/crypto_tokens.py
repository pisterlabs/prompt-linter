"""
NOTE:
- To build the index for crypo tokens, first download the large list from "https://api.coingecko.com/api/v3/coins/list" and save it in "knowledge_base/crypto_tokens.json"
- Then run - python3 -c "from index import crypto_tokens; crypto_tokens.backfill()"
"""

from typing import Any, Iterable, List, Optional
import os
import traceback
import json
import time

from langchain.docstore.document import Document
from .weaviate import get_client


INDEX_NAME = 'CryptoTokensV1'
INDEX_DESCRIPTION = "Index of Crypto Tokens"
CANONICAL_ID_KEY = 'canonical_id'
SYMBOL_KEY = "symbol"
NAME_KEY = "name"



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
                        "description": "The canonical ID of the crypto token",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": CANONICAL_ID_KEY,
                    },
                    {
                        "dataType": ["text"],
                        "description": "The name of the crypto token",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": NAME_KEY,
                    },
                    {
                        "dataType": ["text"],
                        "description": "The symbol of the crypto token",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": SYMBOL_KEY,
                    },
                ],
            },
        ]
    }
    client.schema.create(schema)


def backfill():
    # TODO: right now we don't have stable document IDs unlike sites.
    # Always drop and recreate first.
    from langchain.vectorstores import Weaviate

    with open('./knowledge_base/crypto_tokens.json') as f:
        crypto_tokens = json.load(f)
        documents = [c.pop("id") for c in crypto_tokens]

    create_schema(delete_first=True)

    client = get_client()
    w = Weaviate(client, INDEX_NAME, CANONICAL_ID_KEY)

    # NOTE: OpenAI has API limit of 3000/min so batch process the creation
    batch_size = 40
    for i in range(0, len(documents), batch_size):
        length = i + batch_size
        print("index: ", i, " length:", length)
        w.add_texts(documents[i:length], crypto_tokens[i:length])
        time.sleep(1)


def get_index_size():
    client = get_client()
    data = client.query.aggregate(INDEX_NAME).with_fields('meta { count }').do()
    print(data)
