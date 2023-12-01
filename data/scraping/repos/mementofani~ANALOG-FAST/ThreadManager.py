import time

import openai
from openai import OpenAI

api_key = "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"
client = OpenAI(api_key=api_key)


class ThreadManager:

    def __init__(self, assistant):
        self.client = client
        self.assistant = assistant

    def create_thread(self):
        return self.client.beta.threads.create().id

    def append_message(self, thread_id, content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        ).id

    def get_latest_message(self, thread_id, last_message_id):
        response = self.client.beta.threads.messages.list(thread_id=thread_id)
        for message in reversed(response.data):
            if message.id != last_message_id:
                return message
        return None


def ask_question(self, question) -> str:
    thread_id = self.create_thread()
    last_message_id = self.append_message(thread_id, question)

    client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=self.assistant.id
    )

    while True:
        new_message = self.get_latest_message(thread_id, last_message_id)
        if new_message:
            return new_message.content[0].text.value
        time.sleep(0.5)
