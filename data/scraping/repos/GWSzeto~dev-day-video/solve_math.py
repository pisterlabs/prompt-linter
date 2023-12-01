from openai import OpenAI
from dotenv import load_dotenv
from time import sleep
import os
load_dotenv()

# Create an instance of OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create an assistant
# Speciazlies in solving math problems using python code
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

# This is the channel that the user and assistant will communicate through
thread = client.beta.threads.create()

# User sends a message to the assistant
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Please help me solve the following equation `3x + 11 = 14`"
)

# assistant processes the message sent by the user
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)

# checks the status of the response from the assistant
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    sleep(2)

# # Shows Steps taken by the assistant to solve the math problem
# steps = client.beta.threads.runs.steps.list(
#     thread_id=thread.id,
#     run_id=run.id,
# )
# print(steps.model_dump_json())

# Shows the interaction of messages between the user and the assistant
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
)
print(messages.model_dump_json())

