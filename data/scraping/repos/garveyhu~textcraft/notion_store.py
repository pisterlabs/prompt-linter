from langchain.document_loaders import NotionDBLoader

from textcraft.pinecone_store.pinecone_store import store_document

NOTION_TOKEN = "secret_7Do5qTlmWC9080cl4VaZpz6sCydgRjtuTfPWd8g7zpL"
NOTION_DATABASE_ID = "3fd1353da8d841fd8543b394e85dc84d"

loader = NotionDBLoader(
    integration_token=NOTION_TOKEN,
    database_id=NOTION_DATABASE_ID,
    request_timeout_sec=30,  # optional, defaults to 10
)


def clean_metadata(document):
    for key, value in document.metadata.items():
        if value is None:
            document.metadata[key] = "None"
    return document


docs = loader.load()
cleaned_docs = [clean_metadata(doc) for doc in docs]
store_document(cleaned_docs)
