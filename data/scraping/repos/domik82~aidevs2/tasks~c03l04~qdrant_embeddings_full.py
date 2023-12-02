import json
import os
import uuid

from dotenv import load_dotenv, find_dotenv
from icecream import ic
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct, CollectionStatus, UpdateStatus
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from typing import List

import openai
from openai.embeddings_utils import get_embedding

from dotenv import load_dotenv, find_dotenv

from tasks.c03l04.unknow_database_storage import create_postgresql_session, UnknowNews, delete_sql_data

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


class QdrantVectorStore:

    def __init__(self,
                 host: str = None,
                 port: int = None,
                 collection_name: str = None,
                 vector_size: int = None,
                 vector_distance=Distance.COSINE
                 ):

        self.client = QdrantClient(
            url=host,
            port=port,
            # path=db_path
        )
        self.collection_name = collection_name

        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            ic("Collection does not exist, creating collection now")
            self.set_up_collection(collection_name, vector_size, vector_distance)

    def set_up_collection(self, collection_name: str, vector_size: int, vector_distance: str):

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=vector_distance)
        )

        collection_info = self.client.get_collection(collection_name=collection_name)

    def search_using_embedded_query(self, input_query: str, limit: int = 3):
        input_vector = get_embedding(input_query, engine="text-embedding-ada-002")
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=input_vector,
            limit=limit
        )

        result = []
        for item in search_result:
            similarity_score = item.score
            payload = item.payload
            data = {"id": item.id,
                    "similarity_score": similarity_score,
                    "url": payload.get("url"),
                    "title": payload.get("title")
                    }
            ic(data)
            result.append(data)

        return result

    def delete_record(self, text_ids: list):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=text_ids,
            )
        )

    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name=collection_name)


# Sample JSON data
unknown_json_data = [
    {
        "title": "Niesamowite \"Roboty\" sprzed setek lat - jak to dzia\u0142a\u0142o? (film, 1h)",
        "url": "https:\/\/www.youtube.com\/watch?v=6Nt7xLAfEPs",
        "info": "INFO: Z pewno\u015bci\u0105 znasz figurki poruszaj\u0105ce si\u0119, na przyk\u0142ad w dawnych szopkach bo\u017conarodzeniowych. A mo\u017ce uczy\u0142 Ci si\u0119 'Mechaniczny Turek', kt\u00f3ry ogrywa\u0142 wszystkich w szachy? S\u0105 to urz\u0105dzenia sprzed setek lat. Z filmu dowiesz si\u0119, co wprawia\u0142o te mechanizmy w ruch.",
        "date": "2023-11-10"
    },
    {
        "title": "Ostatni event Apple nagrano iPhonem 15 Pro Max - czy to wa\u017cne dla klienta?",
        "url": "https:\/\/prolost.com\/blog\/scarybts",
        "info": "INFO: Prawd\u0105 jest, \u017ce wydarzenie zosta\u0142o od pocz\u0105tku do ko\u0144ca zrealizowane z u\u017cyciem najnowszego iPhone'a i zmontowane na Macu, ale jakie realnie ma to znaczenie dla klienta ko\u0144cowego? Mo\u017ce wydawa\u0107 si\u0119, \u017ce taki komunikat m\u00f3wi: \"Ty te\u017c mo\u017cesz to osi\u0105gn\u0105\u0107!\", ale czy na pewno tak jest? Ile z \"filmowania iPhone'em\" to marketingowe gadanie, a ile to fakty istotne dla u\u017cytkownika iPhone'a?",
        "date": "2023-11-10"
    },
]


def insert_data(json_data, sql_session, vector_db):
    i = 0
    for item in json_data:
        element_id = str(uuid.uuid4())  # Assign a unique ID for each point
        # Create embeddings for the title and info
        title = item['title']
        info = item['info']
        url = item['url']
        info_date = item['date']
        # title_embedding = create_embeddings(title)

        text_vector = get_embedding(info, engine="text-embedding-ada-002")
        # Combine title and info embeddings
        # combined_embedding = title_embedding + info_embedding

        # Define the metadata
        metadata = {
            "url": url,
            "date": info_date,
            "title": title
        }

        operation_info = vector_db.client.upsert(
            collection_name=qdrant_collection_name,
            points=[PointStruct(
                id=element_id,
                vector=text_vector,
                payload=metadata
            )]
        )
        if operation_info.status == UpdateStatus.COMPLETED:
            ic(f"Data with ID {element_id} has been stored in Qdrant.")
            news = UnknowNews(id=element_id, title=title, url=url, info=info, date=info_date)
            sql_session.add(news)
            sql_session.commit()
            ic(f"Data with ID {element_id} has been stored in postgres.")
            i = i + 1
        else:
            ic("Failed to insert data")

        if i == 300:
            ic("300 rekordów wystarczy do zadania")
            break

    total_count = sql_session.query(UnknowNews).count()
    ic(f'DB count: {total_count}')

    vector_all = vector_db_obj.client.count(collection_name=vector_db_obj.collection_name)
    ic(f"count:{vector_all.count}")


def delete_all(sql_session, sql_table, qdrant_client, qdrant_collection):
    sql_session.query(sql_table).delete()
    # commit the transaction
    sql_session.commit()

    qdrant_client.delete_collection(qdrant_collection)


def load_archive():
    with open('./archiwum.json', 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # quick and dirty :(

    # SQL session
    sql_session_obj = create_postgresql_session()

    # Your Qdrant host and port
    qdrant_host = 'localhost'
    qdrant_port = 6333
    qdrant_collection_name = 'unknow_news'
    ai_vector_size = 1536

    vector_db_obj = QdrantVectorStore(host=qdrant_host,
                                      port=qdrant_port,
                                      collection_name=qdrant_collection_name,
                                      vector_size=ai_vector_size)

    # data_to_process = unknown_json_data

    # data_to_process = load_archive()
    # insert_data(data_to_process, sql_session_obj, vector_db_obj)

    count = sql_session_obj.query(UnknowNews).count()
    ic(f'DB count: {count}')

    response = vector_db_obj.client.count(collection_name=vector_db_obj.collection_name)
    ic(f"count:{response.count}")

    simple_query = "roboty"
    results_query = vector_db_obj.search_using_embedded_query(simple_query)
    ic(results_query)

    question = 'Co różni pseudonimizację od anonimizowania danych?'
    results_query = vector_db_obj.search_using_embedded_query(question)
    ic(results_query)

    # remember not to delete it!
    # delete_all(sql_session_obj, UnknowNews, vector_db_obj, vector_db_obj.collection_name)
