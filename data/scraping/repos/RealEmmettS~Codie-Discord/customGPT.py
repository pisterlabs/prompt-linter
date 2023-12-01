from openai import OpenAI
import json
import time
import os
from replit import db

client = OpenAI()
assistantID = os.getenv("ASSISTANT_ID")

# Generate's a reponse from Codie
def generate_reply(usr_message, threadID, authorID):
    assistant_response = ""
    run = None

    # If no pre-existing conversation, create a new one
    if threadID == "NIL":
        print(f"Creating new conversation with ID: {authorID}")
        run = client.beta.threads.create_and_run(
            assistant_id=assistantID,
            thread={"messages": [{"role": "user", "content": usr_message}]}
        )
    # If pre-existing conversation, append to existing one
    else:
        print(f"Appending to existing conversation with ID: {authorID}")
        message = client.beta.threads.messages.create(
            thread_id=threadID,
            role="user",
            content=usr_message
        )
        run = client.beta.threads.runs.create(
            thread_id=threadID,
            assistant_id=assistantID
        )

    # Polling mechanism to see if runStatus is completed
    while True:
        try:
            retrievedRun = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
            if retrievedRun.status == "completed":
                db[authorID] = retrievedRun.thread_id
                break
        except Exception as e:
            print(f"Encountered an error: {e}")
            # Decide how you want to handle the error here
            # For example, you might want to retry or exit the loop after too many failures
        time.sleep(2)  # Wait for 2 seconds before checking again

    # Get the last assistant message from the array
    messages = client.beta.threads.messages.list(thread_id=run.thread_id)

    for msg in messages.data:
        role = msg.role
        if role == "assistant":
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")
            print("\n--SENDING MESSAGE--\n\n")
            assistant_response = content
            break
    
    return assistant_response