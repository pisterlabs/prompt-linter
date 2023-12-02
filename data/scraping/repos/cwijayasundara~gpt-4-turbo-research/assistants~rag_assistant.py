import openai
import os
import time
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

file_path = "docs/2023q3-alphabet-earnings-release.pdf"

file = client.files.create(
    file=open(file_path, "rb"),
    purpose='assistants'
)

st.header("GPT 4 Retriever Assistants")
user_message = st.text_input("Pls enter your question:")
submit = st.button("submit", type="primary")

if submit:

    # You can attach a maximum of 20 files per Assistant, and they can be at most 512 MB each.
    assistant = client.beta.assistants.create(
        name="RAG Assistant",
        description="You are great at extracting information from documents. You analyze data present in .pdf files, "
                    "understand trends, and come up with answers relevant to those documents. You also share a "
                    "brief text summary of the documents you analyze.",
        model="gpt-4-1106-preview",
        tools=[{"type": "code_interpreter"}],
        file_ids=[file.id]
    )

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Answer questions based on the content of the document.",
                "file_ids": [file.id]
            }
        ]
    )

    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        model="gpt-4-1106-preview",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
    )

    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    time.sleep(20)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id
    )

    if run_steps.data[0].status == "completed":
        print("The run was successful!")
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        for message in messages.data:
            print(message.role.upper(), message.content[0].text.value)
            role = message.role.upper()
            content = message.content[0].text.value
            st.write(role, " : ", content)
