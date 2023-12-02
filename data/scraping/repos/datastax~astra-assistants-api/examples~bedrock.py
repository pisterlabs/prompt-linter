import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./.env")


# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY="Fake key"
ASTRA_DB_TOKEN=os.getenv("ASTRA_DB_TOKEN")
BEDROCK_AWS_REGION=os.getenv("BEDROCK_AWS_REGION")
BEDROCK_AWS_ACCESS_KEY_ID=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID")
BEDROCK_AWS_SECRET_ACCESS_KEY=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY")
base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "embedding-model": "amazon.titan-embed-text-v1",
        "LLM-PARAM-aws-access-key-id": BEDROCK_AWS_ACCESS_KEY_ID,
        "LLM-PARAM-aws-secret-access-key": BEDROCK_AWS_SECRET_ACCESS_KEY,
        "LLM-PARAM-aws-region-name": BEDROCK_AWS_REGION,
    }
)

thread = client.beta.threads.create()
my_thread = client.beta.threads.retrieve(thread.id)
updated = client.beta.threads.update(thread.id, metadata={"hi": "there"})

client.beta.threads.messages.create(thread_id=thread.id, content="some content", role="user")
deleted = client.beta.threads.delete(thread.id)
print(my_thread)

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="meta.llama2-13b-chat-v1",
)

print(assistant)


print("Uploading file:")

# Upload the file
file = client.files.create(
    file=open(
        "./examples/language_models_are_unsupervised_multitask_learners.pdf",
        "rb",
    ),
    purpose="assistants",
)

# Update Assistant
assistant = client.beta.assistants.update(
    assistant.id,
    tools=[{"type": "retrieval"}],
    file_ids=[file.id],
)

print(assistant)


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def create_thread_and_run(user_input, assistant_id):
    thread = client.beta.threads.create()
    run = submit_message(assistant_id, thread, user_input)
    return thread, run


thread, run = create_thread_and_run(
    "What are some cool math concepts behind this ML paper pdf? Explain in two sentences.",
    assistant.id
)
def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="desc")

def pretty_print(messages):
    print("# Messages")
    for m in messages.data:
        print(f"{m.role}: {m.content[0].text['value']}")
    print()


# Waiting in a loop
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

run = wait_on_run(run, thread)
pretty_print(get_response(thread))
