"""scraibe/bot.py"""

import hashlib
from scraibe.store import Vector, VectorStore
from config import DATA_DIR
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import openai


class Bot:
    def __init__(self, query: str = None, clinical_test: str = None):
        self.query = query

        self.vector_file_path = f"{DATA_DIR}/vector.txt"
        self.vector_store = VectorStore()

        self.query_embeddings = self.get_embedded(query)
        self.query_vector = Vector(text=query, embedding=self.query_embeddings)

    @staticmethod
    def get_embedded(text: str):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text,
        )
        embeddings = response["data"][0]["embedding"]
        return embeddings

    @staticmethod
    def chunk_text(text: str) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.create_documents([text])

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def add_to_vector_store(self, text: str):
        if self._is_text_in_vector_file(text):
            print("Text already exists in the vector store.")
            return
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.vector_store.push(Vector(chunk))
        self._add_text_to_vector_file(text)

    def _is_text_in_vector_file(self, text: str) -> bool:
        hash_value = self._hash_text(text)
        with open(self.vector_file_path, "r") as f:
            for line in f.readlines():
                if hash_value in line:
                    return True
        return False

    def _add_text_to_vector_file(self, text: str):
        hash_value = self._hash_text(text)
        with open(self.vector_file_path, "a") as f:
            f.write(f"{hash_value}\n")

    def get_similar_texts(self):
        # Assuming VectorStore has a method to retrieve similar vectors
        similar_vectors = self.vector_store.select_nearest(Vector(self.query_vector))
        return [vector.document for vector in similar_vectors]

    def chat(self):
        # Starting with an initial system message
        messages = [
            {
                "role": "system",
                "content": "You are a medical assistant with access to historical medical notes.",
            }
        ]

        while True:
            # Taking user input
            self.query = input("You: ")

            # Exit condition
            if self.query.lower() in ["exit", "bye"]:
                print("Bot: Goodbye!")
                break

            # Fetching similar texts using vector embeddings
            similar_texts = self.get_similar_texts()

            # Appending the extracted medical notes to the conversation history
            for text in similar_texts:
                messages.append({"role": "assistant", "content": f"Note: {text}"})

            # Append user's message to the list
            messages.append({"role": "user", "content": self.query})

            # Requesting the model for a response
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
            )

            # Extracting the model's response
            assistant_response = response["choices"][0]["message"]["content"]

            # Printing and appending the model's response to the list of messages
            print(f"Bot: {assistant_response}")
            messages.append({"role": "assistant", "content": assistant_response})
