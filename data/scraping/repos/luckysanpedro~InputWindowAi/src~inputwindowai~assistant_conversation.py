from dotenv import load_dotenv
from openai import OpenAI
import os

# Load your OpenAI API key
load_dotenv()
client = OpenAI()

# for the assistant:
initial_message = "Hello"
assistant_id = os.getenv("AssID")
thread = None
user_input = ""


def start_interact_with_assistant(client, assistant_id, initial_message):
    global assistant_message
    # Create a thread with the initial message
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=initial_message
    )
    return thread


def interact_with_assistant(client, assistant_id, thread, user_input):
    # Ask the user for input
    global assistant_message

    print("User: ", user_input)
    # Add the user's message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_input
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    # Wait for the run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Retrieve and print the last assistant message
    message = client.beta.threads.messages.list(thread_id=thread.id).data[0]
    if message.role == "assistant":
        assistant_message = message.content[0].text.value
        print("Assistant: ", assistant_message)

    return assistant_message


def initialize():
    global thread
    thread = start_interact_with_assistant(client, assistant_id, initial_message)


def run_blocking_operations(user_input, assistant_id):
    global thread
    if thread is None:
        initialize()
    assistant_message = interact_with_assistant(
        client, assistant_id, thread, user_input
    )
    return assistant_message


def main(user_input, assistant_id):
    # Run the blocking operations
    assistant_message = run_blocking_operations(user_input)
    print("assistant_message:", assistant_message)
    return assistant_message


if __name__ == "__main__":
    main(user_input)
