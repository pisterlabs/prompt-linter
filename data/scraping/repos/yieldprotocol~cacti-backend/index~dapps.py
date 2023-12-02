# to build the index for dapps, first scrape them using the scraper
# then run: python3 -c "from index.dapps import backfill; backfill()"


from langchain.docstore.document import Document
from .weaviate import get_client
import json

INDEX_NAME = "Web3Apps"
INDEX_DESCRIPTION = "Index of Third party dapps"
DAPP_DESCRIPTION = "description"
DAPP_NAME = "name"
DAPP_URL = "url"

def delete_schema() -> None:
    try: 
        client = get_client()
        client.schema.delete_class(INDEX_NAME)
    except Exception as e: 
        print(f"Error deleting schmea: {str(e)}")

def create_schema(delete_first: bool = False) -> None:
    try: 
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
                            "type": "text"
                        }
                    },
                    "properties": [
                        {"name": DAPP_NAME, "dataType": ["text"]},
                        {"name": DAPP_DESCRIPTION, "dataType": ["text"]},
                        {
                            "name": DAPP_URL, 
                            "dataType": ["text"],
                            "description": "The URL of the Dapp",
                            "moduleConfig": {
                                    "text2vec-openai": {
                                    "skip": True,
                                    "vectorizePropertyName": False
                                }
                            }
                        },
                    ]
                }
        
            ]
        }
        client.schema.create(schema)
    except Exception as e:
        print(f"Error creating schema: {str(e)}")

def backfill():
    try: 
        from langchain.vectorstores import Weaviate

        with open('./knowledge_base/dapps_ranked_unique.json') as f: 
            dapp_list = json.load(f)
            
        documents = [d.get("description") for d in dapp_list]

        metadatas = dapp_list
            
        create_schema(delete_first=True)

        client = get_client()
        w = Weaviate(client, INDEX_NAME, DAPP_NAME) 
        w.add_texts(documents, metadatas)
    except Exception as e: 
        print(f"Error during backfill in dapps.py {str(e)}")


