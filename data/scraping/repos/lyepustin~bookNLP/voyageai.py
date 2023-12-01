 # File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description: 

from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
import qdrant_client
import ebooklib
import logging
import os
import sys
import time
import requests
import json
import chardet

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
load_dotenv()

from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

def recreate_qdrant_collection(collection_name, size):

    client = get_qdrant_client()
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logging.info(f"'{collection_name}' collection re-created.")
    except Exception as e:
        logging.error(
            f"on create collection '{collection_name}'. " + str(e).replace('\n', ' '))


def get_text_chunks(text, max_chunk_size=os.getenv("TEXT_SPLITTER_CHUNK_SIZE")):
    text_splitter = CharacterTextSplitter(
        separator=str(os.getenv("TEXT_SPLITTER_SEPARATOR")),
        chunk_size=int(max_chunk_size),
        chunk_overlap=int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP")),
        length_function=len
    )
    chunks = text_splitter.split_text(str(text))
    return chunks


def get_qdrant_client():
    return qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        https=True)


def read_book_sample():
    return os.getenv("TEXT_SAMPLE")


def read_book(ebook_path):
    import warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="ebooklib.*")
    raw_text_list = []
    for item in epub.read_epub(ebook_path, {"ignore_ncx": False}).get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            raw_content = item.get_body_content().decode('utf-8')
            soup = BeautifulSoup(raw_content, "html.parser")
            paragraphs = soup.find_all("p")
            for paragraph in paragraphs:
                raw_text_list.append(paragraph.get_text())
    return get_text_chunks(" ".join(raw_text_list))
    

class VoyageAI:
    def __init__(self):
        self.api_key = os.getenv("VOYAGEAI_API_KEY")
        self.api_url = "https://api.voyageai.com/v1/embeddings"
        self.model = os.getenv("VOYAGEAI_MODEL")

    def get_free_api_usage(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "input": ["test"],
            "model": self.model
        }

        response = requests.post(self.api_url, json=data, headers=headers)
        if response.status_code == 200:
            prompt_tokens = response.json().get("usage").get("prompt_tokens")
            total_tokens = response.json().get("usage").get("total_tokens")
            max_tokens = 50000000
            used_prompt_tokens = (prompt_tokens * 100) / max_tokens
            used_total_tokens = (total_tokens * 100) / max_tokens
            if used_total_tokens == used_prompt_tokens:
                return f"[VoyageAI] Used {used_total_tokens:.3f}% of free usage"
            else:
                return f"[VoyageAI] Used {used_prompt_tokens:.3f}% of free usage for prompt tokens and {used_total_tokens:.3f}% for total tokens"
        else:
            return f"[VoyageAI] Error: {response.status_code}, {response.text}"

    def get_embeddings(self, texts):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "input": texts,
            "model": self.model
        }

        return requests.post(self.api_url, json=data, headers=headers)


class ChatbotOpenAI():
    """
    Class to interact with OpenAI for responses.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo"
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.max_retry = 5
        self.request_timeout = 20


    def answer_question(self, question, context):
        """
        Chat models take a list of messages as input and return a model-generated message as output.
        """
        retry = 0
        self.prompt = f"Contesta la siguiente pregunta:'{question}' con el contexto proporcionado por el usuario."
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.model,
                        "temperature": 0.5,
                        "messages": [
                            {"role": "system", "content": self.prompt},
                            {"role": "user", "content": str(context)},
                        ],
                    },
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except (requests.exceptions.RequestException, KeyError) as oops:
                retry += 1
                if retry >= self.max_retry:
                    return "OpenAI error: %s" % oops
                logging.getLogger(__name__).error("Error communicating with OpenAI: ", oops)
                time.sleep(1)


def add_some_text():
    # recreate_qdrant_collection(
    #     os.getenv("VOYAGEAI_MODEL"), os.getenv("VOYAGEAI_EMBEDDING_DIM_SIZE"))

    content = read_book(os.getenv("BOOK_PATH")) 
    text_chunks = get_text_chunks(
        content, max_chunk_size=os.getenv("VOYAGEAI_MAX_CONTEXT_LENGHT"))
    
    voyage = VoyageAI()
    print(voyage.get_free_api_usage())


    for i in range(0, len(text_chunks), int(os.getenv("VOYAGEAI_MAX_CONTEXT_BY_REQUEST"))):
        sub_chunks = text_chunks[i:i+int(os.getenv("VOYAGEAI_MAX_CONTEXT_BY_REQUEST"))]
        response = voyage.get_embeddings(sub_chunks)
        
        if response.status_code == 200:
            client = get_qdrant_client()
            a = client.upsert(
                collection_name=os.getenv("VOYAGEAI_MODEL"),
                points=[
                    PointStruct(
                        id=vector.get("index"),
                        vector=vector.get("embedding"),
                        payload={"text": text_chunks[vector.get("index")]}
                    )
                    for vector in response.json().get("data")
                ]
            )
            print(a)
            
        else:
            print(f"Error: {response.status_code}, {response.text}")
            # retry?

    print(voyage.get_free_api_usage())

def make_some_query():

    question = "Cuantos ejemplares del libro maldito es La Divina Comedia ha descubierto Victor Ros?"
    voyage = VoyageAI()
    response = voyage.get_embeddings(question).json()
    
    if "data" not in response:
        raise RuntimeError(f"Voyage API Error. Message: {json.dumps(response)}")

    client = get_qdrant_client()
    hits = client.search(
        collection_name=os.getenv("VOYAGEAI_MODEL"),
        query_vector=response.get("data")[0].get("embedding"),
        limit=3  # Return 5 closest points
    )
    
    context = ""
    for hit in hits:
        context += hit.payload.get("text") + " "

    chatbot = ChatbotOpenAI()
    response = chatbot.answer_question(question, context)
    print(response)
            
         
if __name__ == '__main__':
    # add_some_text()
    make_some_query()