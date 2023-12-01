import datetime

from langchain.memory.chat_message_histories import (
    FirestoreChatMessageHistory as _FirestoreChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, messages_from_dict
from google.cloud.firestore import Client, CollectionReference
from firebase_admin import firestore


class FirestoreChatMessageHistory(_FirestoreChatMessageHistory):
    _collection: CollectionReference = None

    def prepare_firestore(self) -> None:
        # Prepare the Firestore client
        self.firestore_client: Client = firestore.client()

        # Create a reference to the collection for this user and session
        self._collection = self.firestore_client.collection(
            f"{self.collection_name}/{self.user_id}/{self.session_id}"
        )

        # Load the messages from the database, called once when the history is created
        self.load_messages()

    def load_messages(self) -> None:
        count = self._collection.count().get()
        if len(count) > 0:
            docs = self._collection.order_by("timestamp", direction="DESCENDING").get()
            self.messages = messages_from_dict([doc.to_dict() for doc in docs])

    def add_message(self, message: BaseMessage) -> None:
        # Add the message to the in-memory list
        self.messages.append(message)

        # Persist the message to the database
        self.firestore_client.collection(
            f"{self.collection_name}/{self.user_id}/{self.session_id}"
        ).add(
            {
                "data": message.dict(),
                "type": message.type,
                "timestamp": datetime.datetime.now(),
            }
        )

    def clear(self) -> None:
        if not self._collection:
            raise ValueError("Collection not initialized!")

        batch = self.firestore_client.batch()
        docs = self._collection.list_documents(page_size=500)

        # Delete documents in chunks
        for doc in docs:
            batch.delete(doc)

        batch.commit()
