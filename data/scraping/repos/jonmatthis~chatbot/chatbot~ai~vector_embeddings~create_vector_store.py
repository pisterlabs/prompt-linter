import asyncio
import os
from copy import deepcopy
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from chatbot.ai.vector_embeddings.plot_vector_clusters import visualize_clusters
from chatbot.mongo_database.mongo_database_manager import MongoDatabaseManager
from chatbot.system.filenames_and_paths import get_thread_backups_collection_name


async def get_or_create_student_message_vector_store(
        thread_collection_name: str = get_thread_backups_collection_name()):
    mongo_database = MongoDatabaseManager()
    collection = mongo_database.get_collection(thread_collection_name)
    all_thread_entries = await collection.find().to_list(length=None)

    print(f"Creating document list from {thread_collection_name} collection with {len(all_thread_entries)} entries")

    student_vector_stores = {}
    student_documents = {}
    for thread_entry in all_thread_entries:
        student_initials = thread_entry["_student_initials"]
        if student_initials not in student_documents:
            student_documents[student_initials] = []

        list_of_strings = thread_entry["thread_as_list_of_strings"]
        print(f"Creating vector store for student: {student_initials}  with {len(list_of_strings)} chunks")
        for message in thread_entry["messages"]:

            reactions = ""
            if len(message["reactions"]) > 0:
                reactions = ",".join([f"{reaction}" for reaction in message["reactions"]])

            metadata = {"message_type": "human" if message["human"] else "bot",
                        "_student_name": student_initials,
                        "_student_uuid": thread_entry["_student_uuid"],
                        "server_name": thread_entry["server_name"],
                        "channel_name": thread_entry["channel"],
                        "thread_id": thread_entry["thread_id"],
                        "thread_url": thread_entry["thread_url"],
                        "message_url": message["jump_url"],
                        "thread_as_one_string": thread_entry["thread_as_one_string"],
                        "created_at": message["created_at"].isoformat(),
                        "reactions": reactions,
                        }

            student_documents[student_initials].append(Document(page_content=message["content"], metadata=metadata))

    print("------------------------------------\n",
          "------------------------------------\n")

    vectorstore_collection_name = "student_vector_store"
    for student_initials, student_message_documents in student_documents.items():
        print(f"Creating vector store from {student_initials} collection with {len(student_message_documents)} entries")
        embeddings_model = OpenAIEmbeddings()
        persistence_directory = str(Path(os.getenv("PATH_TO_CHROMA_PERSISTENCE_FOLDER")) / vectorstore_collection_name)
        student_vector_stores[student_initials] = Chroma.from_documents(
            documents=student_message_documents,
            embedding=embeddings_model,
            collection_name=vectorstore_collection_name,
            persist_directory=persistence_directory,
        )

    return student_vector_stores


def chunk_list_of_strings(list_of_strings):
    def chunk_string_list(lst):
        return [lst[i - 1:i + 2] for i in range(1, len(lst) - 1)]

    chunks = chunk_string_list(list_of_strings)
    return chunks


async def create_green_check_vector_store(collection_name: str = "green_check_messages"):
    mongo_database = MongoDatabaseManager()
    collection = mongo_database.get_collection(collection_name)
    all_entries = await collection.find().to_list(length=None)
    print("Creating vector store from {collection_name} collection with {len(all_entries)} entries")
    documents = []
    for entry in all_entries:
        entry.pop("_id")
        entry.pop("green_check_messages")

        documents.append(Document(page_content=entry["abstract"],
                                  metadata={"source": entry["citation"],
                                            **entry}))


    chroma_vector_store = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        collection_name=collection_name,
        # persist_directory=str(Path(os.getenv("PATH_TO_CHROMA_PERSISTENCE_FOLDER")) / collection_name),
    )
    return chroma_vector_store


def split_string(s, length, splitter: str = "<br>") -> str:
    return splitter.join(s[i:i + length] for i in range(0, len(s), length))


async def main():
    vector_store = await create_green_check_vector_store()
    collection = vector_store._collection.get(include=["embeddings", "documents", "metadatas"])
    # vector_stores = await get_or_create_student_message_vector_store()
    # embeddings = []

    # for student_name, vector_store in vector_stores.items():
    #     collection = vector_store._collection.get(include=["embeddings", "documents", "metadatas"])
    #     embeddings.extend(collection["embeddings"])
    #     labels.extend([split_string(document, 30) for document in collection["documents"]])
    labels = []
    for metadata in collection["metadatas"]:
        labels.append(f"{metadata['_student_initials']} - {metadata['source']}")
    visualize_clusters(embeddings=collection["embeddings"], labels=labels, n_clusters=5)


if __name__ == "__main__":
    asyncio.run(main())
