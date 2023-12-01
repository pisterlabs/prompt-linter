import os
import csv
import openai
import chromadb

from dotenv import load_dotenv, find_dotenv

from chromadb.utils import embedding_functions

_ = load_dotenv(find_dotenv(filename="path/to/your/.env/file"))

openai.api_key = os.environ['OPENAI_API_KEY']

def read_csv_file(file_path):
    data_list = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            data_list.append(row)

    return data_list

def get_collection():
    file_path = "path/to/your/csv/file"
    data = read_csv_file(file_path)[1:]

    idx = []
    headers = []
    descriptions = []
    times = []
    levels = []
    for row in data:
        idx.append(row[0])
        headers.append(row[1])
        descriptions.append(row[2])
        times.append(row[5])
        levels.append(row[7])

    metadata = []
    for i in range(len(data)):
        metadata_dict = {"header" : headers[i], "time" : times[i], "level" : levels[i]}
        metadata.append(metadata_dict)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai.api_key,
                    model_name="text-embedding-ada-002"
                )

    client = chromadb.PersistentClient(path="./")

    collection = client.create_collection(
        name="collection",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=descriptions,
        metadatas=metadata,
        ids=idx
    )

    return collection
