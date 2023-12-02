import os
import re
import threading
import uuid

from dotenv import load_dotenv

load_dotenv()
import openai
from firebase_admin import firestore

openai.api_key = os.getenv("OPENAI_API_KEY")

prefix = re.compile(r"^(ML Tutor|You):\s*")


# Watch the collection query


class MessageService:
    def __init__(self, db: firestore.Client):
        self.db = db
        self.collection = db.collection("messages")
        self.model = "text-davinci-003"
        self.callback_done = threading.Event()
        self.messages = self.get_messages(self.collection.stream())
        self.collection.on_snapshot(self.on_snapshot)

    def on_snapshot(self, snapshot, changes, real_time):
        self.messages = self.get_messages(snapshot)
        print("message updated")
        self.callback_done.set()

    def reply(self, prompt: str):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=0.3,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"],
        )
        return response["choices"][0]["text"].strip()

    def get_messages(self, docs):
        docs = [doc.to_dict() for doc in docs]
        docs.sort(key=lambda doc: doc["timestamp"])
        return docs

    def add_message(self, text: str, received: bool):
        return self.collection.document().set(
            {
                "id": str(uuid.uuid4()),
                "text": re.sub(prefix, "", text),
                "received": received,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
