from datetime import datetime

import lancedb
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore, LanceDB

from config import Config
from utils.files import get_root_path


def get_vectorstore(table_name: str, embedding: Embeddings) -> VectorStore:
    config = Config()
    db_path = get_root_path() / config.lancedb_url
    db = lancedb.connect(db_path)
    if not table_name in db.table_names():
        table = db.create_table(table_name, data=[
            {
                "vector": embedding.embed_query("Hello World"),
                "text": "Hello World",
                "url": "https://google.com/",
                "time": datetime.now().timestamp()}
        ])
    else:
        table = db.open_table(table_name)

    vectorstore = LanceDB(embedding=embedding, connection=table)

    return vectorstore