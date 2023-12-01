import os
from urllib.parse import quote_plus

from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from get_data_for_db import get_asian_documents, get_mexican_documents


def main():
    load_dotenv(find_dotenv())
    # Connection parameters
    username = quote_plus(os.getenv("MONGO_GMATE_USERNAME"))
    password = quote_plus(os.getenv("MONGO_GMATE_PASSWORD"))
    cluster = os.getenv("MONGO_GMATE_CLUSTER")
    database_name = os.getenv("MONGO_GMATE_DATABASE")

    uri = f"mongodb+srv://{username}:{password}@{cluster}.q1ds6re.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))
    print(client.admin.command('ping'))
    db = client[database_name]
    restaurants_collection = db["restaurants"]
    embeddings = OpenAIEmbeddings()

    # Make sure the 'restaurants' collection exists in MongoDB Atlas
    # Create the 'vector_geo_index' search index manually in Atlas UI using the `create_vector_geo_index.json` file
    # todo: automatically create the index

    # Upload records into the collection
    index_name = "vector_geo_index"

    docs = get_asian_documents()
    docs.extend(get_mexican_documents())
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        docs, embeddings, collection=restaurants_collection, index_name=index_name
    )

    if "restaurants" in db.list_collection_names():
        vectorstore = MongoDBAtlasVectorSearch(restaurants_collection, embeddings, index_name=index_name)
    else:
        docs = get_asian_documents()
        docs.extend(get_mexican_documents())
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            docs, embeddings, collection=restaurants_collection, index_name=index_name
        )

    # Test similarity search (disabled until $vectorSearch is fixed on MongoDB Atlas side)
    # search_result = vectorstore.similarity_search(query='sushi', k=5)
    # print(list(map(lambda x: search_result[0].metadata['_id'], search_result)))

    search_result = vectorstore.similarity_search(
        query='tacos',
        k=5,
        pre_filter={
            "geoWithin": {
                "circle": {
                    "center": {
                        "type": "Point",
                        "coordinates": [-82.33554135164104, 28.175078895215975]
                    },
                    "radius": 16000
                },
                "path": "location"
            }
        },
        post_filter_pipeline=[{
            "$project": {
                "embedding": 0,
                "location": 0
            }
        }]
    )
    print(list(map(lambda x: search_result[0].metadata['_id'], search_result)))


if __name__ == '__main__':
    load_dotenv('../.env')
    main()
