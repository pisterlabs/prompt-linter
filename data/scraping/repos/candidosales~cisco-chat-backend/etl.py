import csv
import os
import openai

from dotenv import load_dotenv, find_dotenv

from loaders.text import TextLoader
from chunckers.text import TextChunker
from advisory.models import Advisory

from embedchain import App

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]


chat_bot = App()


def read_csv(max_items: int):
    with open("./data/output-2023-06-22T03:37:39.362Z.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader, None)  # Jump the header
        advisories: list[Advisory] = []
        for index, row in enumerate(csvreader):
            if index + 1 == max_items:
                break
            advisories.append(
                Advisory(
                    url=row[0],
                    id=row[1],
                    title=row[2],
                    severity=row[3],
                    cveList=row[4],
                    cvsScore=row[5],
                    summary=row[6],
                    affectedProducts=row[7],
                    firstPublished=row[8],
                    details=row[9],
                    workarounds=row[10],
                    fixedSoftware=row[11],
                    exploitationPublicAnnouncements=row[12],
                    source=row[13],
                )
            )
        return advisories


def add_advisories_db_sync(advisories: list[Advisory]):
    for advisory in advisories:
        embed_advisories(advisory)


def embed_advisories(advisory: Advisory):
    """
    Loads the data from the given URL, chunks it, and adds it to the database.

    :param loader: The loader to use to load the data.
    :param chunker: The chunker to use to chunk the data.
    :param url: The URL where the data is located.
    """

    chunker = TextChunker()
    loader = TextLoader()
    embeddings_data = chunker.create_chunks(loader, advisory)
    documents = embeddings_data["documents"]
    metadatas = embeddings_data["metadatas"]
    ids = embeddings_data["ids"]
    # get existing ids, and discard doc if any common id exist.
    existing_docs = chat_bot.collection.get(
        ids=ids,
        # where={"url": url}
    )
    existing_ids = set(existing_docs["ids"])

    if len(existing_ids):
        data_dict = {
            id: (doc, meta) for id, doc, meta in zip(ids, documents, metadatas)
        }
        data_dict = {
            id: value for id, value in data_dict.items() if id not in existing_ids
        }

        if not data_dict:
            print(f"All data from {advisory.url} already exists in the database.")
            return

        ids = list(data_dict.keys())
        documents, metadatas = zip(*data_dict.values())

    chat_bot.collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Saved {advisory.url}. Total chunks: {chat_bot.collection.count()}\n")


if __name__ == "__main__":
    limit = 4697
    advisories = read_csv(limit)
    if len(advisories) > 0:
        # with Pool() as pool:
        #     result = pool.map(embed_advisories, advisories)
        add_advisories_db_sync(advisories)

    print(f"Successfully saved! Total chunks: {chat_bot.collection.count()} \n")
