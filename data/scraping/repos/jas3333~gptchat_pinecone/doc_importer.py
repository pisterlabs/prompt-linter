import os
import sys
import re
import pymongo
import requests
import pinecone

from uuid import uuid4
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")
pinecone_api = os.getenv("PINECONE_API")
pinecone_environment = os.getenv("PINECONE_ENV")
pinecone_index = os.getenv("pinecone_index")
headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {open_ai_key}"}
embeddings_url = "https://api.openai.com/v1/embeddings"
mongo_db_connection = os.getenv("MONGODB")
mongo_db_name = os.getenv("mongo_db_name")
mongo_collection = 'messages'

# Mongo DB Connection
client = pymongo.MongoClient(mongo_db_connection)

# Pinecone connection
pinecone.init(api_key=str(pinecone_api), environment=str(pinecone_environment))
vector_database = pinecone.Index(str(pinecone_index))


def get_selected_file():
    files = os.listdir("docs/")

    while True:
        if files:
            for index in range(len(files)):
                print(f"[{index + 1}]: {files[index]}  ")
        else:
            print("No files found.")
            sys.exit()

        print("--------------------------------------------------")
        user_input = input(
            "\nWhich file would you like to import? [q to quit]: ")

        try:
            if user_input.lower() == "q":
                sys.exit()
            elif files[int(user_input) - 1]:
                return files[int(user_input) - 1]
            elif int(user_input) > len(files):
                print("File doesn't exist.")
            else:
                print("Not a valid option.")
        except Exception as error:
            print(error)


def splitter(filename):
    loader = UnstructuredPDFLoader(f"docs/{filename}")
    data = loader.load()
    file_title = filename.split(".")[0]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    os.mkdir(f"finished_text/{file_title}")

    for i in range(len(texts)):
        with open(f"finished_text/{file_title}/{filename.split('.')[0]}_{i}.txt", "w") as file:
            print(f"Dumping data into {filename.split('.')[0]}_{i}.txt")
            file.write(str(texts[i]))


def format_texts(filename):
    file_title = filename.split('.')[0]
    files = os.listdir(f"./finished_text/{file_title}")

    for file in files:
        if file.endswith(".txt"):
            with open(f"./finished_text/{file_title}/{file}") as f:
                contents = f.read()

            match = re.search(r"page_content='(.*?)'", contents)

            if match:
                with open(f"./finished_text/{file_title}/{file}", "w") as f:
                    f.write(match.group(1))
                    print(f"Updating text: {match.group(1)}\n")


def get_embeddings(content):
    content = content.encode(encoding="ASCII", errors="ignore").decode()
    data = {"input": content, "model": "text-embedding-ada-002"}
    while True:
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings", headers=headers, json=data)
            data = response.json()
            vector = data["data"][0]["embedding"]
            return vector
        except Exception as error:
            print(f"An exception occured: {error}")
            continue


def get_file_contents(filename):
    documents = []
    file_title = filename.split('.')[0]
    document_files = os.listdir(f"./finished_text/{file_title}")

    for file in document_files:
        if file.endswith(".txt"):
            with open(f"./finished_text/{file_title}/{file}") as f:
                contents = f.read()
                documents.append(contents)

    return documents


def inject(content):
    unique_id = str(uuid4())

    print(content)

    new_document = {
        "_id": unique_id,
        "message": content,
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    }

    # Database injection
    database = client[f"{mongo_db_name}"]
    collection = database.get_collection(f"{mongo_collection}")
    collection.insert_one(new_document)

    # Pinecone injection
    vector = get_embeddings(content)
    try:
        if vector:
            vector_database.upsert(vectors=[(unique_id, vector)])
    except Exception as error:
        print(error)


filename = get_selected_file()

if filename:
    print(
        f"Chunking {filename}, this could take several minutes(yes 20+ minutes) depending on size of the PDF.")
    print("Don't close until it's finished.")

    # Split the text into chunks and save to several .txt files
    splitter(filename)
    # Grabs just the text and re-writes to the same file
    format_texts(filename)
    # Store all the file contents into a list
    content = get_file_contents(filename)
    for x in range(len(content)):
        print(f"Injecting #{x}: {content[x]}\n")
        inject(content[x])
