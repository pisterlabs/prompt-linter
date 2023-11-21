# /temp /test scripts

import json
from collections import deque
from datetime import datetime

class Conversation:
    def __init__(self):
        self.entries = deque(maxlen=1000)  # Set an appropriate maxlen to limit the deque size
        self.load_data_from_file()

    def add_entry(self, entry, speaker):
        new_entry = {
            "user_id": "your_user_id",  # Replace with your actual user ID
            "entry": entry,
            "speaker": speaker,
            "created_at": str(datetime.now()),
        }
        self.entries.append(new_entry)
        self.save_data_to_file()

    def get_conversation(self, limit):
        filtered_entries = (
            entry
            for entry in self.entries
            if entry["user_id"] == "your_user_id"  # Replace with your actual user ID
        )
        sorted_entries = sorted(
            filtered_entries, key=lambda x: datetime.fromisoformat(x["created_at"]), reverse=True
        )
        conversation_data = (
            f"{entry['speaker'].upper()}: {entry['entry']}" for entry in sorted_entries
        )
        return list(conversation_data)[:limit][::-1]

    def save_data_to_file(self):
        with open("conversation_data.json", "w") as f:
            json.dump(list(self.entries), f, indent=2)

    def load_data_from_file(self):
        try:
            with open("conversation_data.json", "r") as f:
                self.entries = deque(json.load(f), maxlen=1000)  # Set the appropriate maxlen

        except FileNotFoundError:
            # If the file doesn't exist (first run), the entries deque will be empty
            self.entries = deque(maxlen=1000)  # Set the appropriate maxlen


# Usage example:
conversation = Conversation()

# Add entries to the conversation
conversation.add_entry("Hello!", "Alice")
conversation.add_entry("Hi there!", "Bob")
conversation.add_entry("Nice weather today.", "Alice")
conversation.add_entry("Indeed, it is.", "Bob")

# Get the conversation history with a limit of 3 entries
history = conversation.get_conversation(limit=3)

print(history)  # Output will be a list of conversation entries





# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
