"""
This module demonstrates how to use the OpenAI Python client to interact with
the OpenAI API. It creates a thread, sends a message to the thread, and then
runs the thread through an assistant. It then prints the messages and citations
from the assistant's response.
"""

import os
import time

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

client = OpenAI(api_key=OPENAI_API_KEY)
assistant = client.beta.assistants.retrieve(assistant_id=OPENAI_ASSISTANT_ID)
thread = client.beta.threads.create()

prompt = input("User: ")

while True:
    if prompt == "exit":
        break

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    while run.status in ("queued", "in_progress"):
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order="asc",
    )

    print("\033c", end="")  # Clear the terminal

    for message in messages:
        text = message.content[0].text
        print(f"{message.role.capitalize()}: {text.value}\n")

        file_citations = [
            annotation
            for annotation in text.annotations
            if annotation.type == "file_citation"
        ]

        if not file_citations:
            continue

        print("Citations:")
        for citation in file_citations:
            print(f"{citation.text}: {citation.file_citation.quote}\n")

    prompt = input("User: ")
