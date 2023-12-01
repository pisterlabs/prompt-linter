import openai
import time

# Initialize the client
client = openai.OpenAI()

file = client.files.create(
    file=open("songs.csv", "rb"),
    purpose='assistants'
)

# Step 1: Create an Assistant
assistant = client.beta.assistants.create(
    name="Data Analyst Assistant",
    instructions="You are a personal Data Analyst Assistant",
    model="gpt-4-1106-preview",
    tools=[{"type": "code_interpreter"}],
    file_ids=[file.id]
)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Step 3: Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What are the most popular tracks and their artists?"
)

# Step 4: Run the Assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="What are the most popular tracks and their artists?"
)

# print(run.model_dump_json(indent=4))

while True:
    # Wait for 5 seconds
    time.sleep(5)

    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    # print(run_status.model_dump_json(indent=4))

    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # Reverse the order of messages to display them chronologically
        messages.data.reverse()

        # Loop through messages and print content based on role
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

        break
    else:
        print("Waiting for the Assistant to process...")
        time.sleep(5)