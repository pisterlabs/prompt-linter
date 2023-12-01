import openai
import os
import time

#####################
# Configuration Section
#####################

# Load the OpenAI API key from the environment variable for security
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client for beta functionality
client = openai.Client()

# Define the Assistant's ID for later usage
assistant_id = "asst_N1P9Wv5HOiJRJKAekqs4dVP7"

# Specify custom HTTP headers needed to access beta features
headers = {"OpenAI-Beta": "assistants=v1"}

# Specify the interval to wait between checks when polling for a response
SLEEP_INTERVAL = 2

#####################
# Assistant Configuration
#####################

# Create the Assistant with predefined configurations and capabilities
assistant = client.beta.assistants.create(
    name="Reese Istor",
    instructions=(
        "As an authority in power electronics theory and design, your guidance is sought in addressing inquiries related to a variety of power applications. Please provide expert advice on design principles, theoretical understanding, troubleshooting techniques, safety protocols, and the latest technological developments in the field. Draw upon the information from the extensive range of books at your disposal to inform your responses."
    ),
    model="gpt-4-1106-preview",
    tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
)

#####################
# Thread Configuration
#####################

# Initialize a thread which represents a conversation between a user and the assistant
thread = client.beta.threads.create(
    messages=[
        {"role": "user", "content": "Please assist me with power electronics."}
    ]
)

#####################
# Utility Functions Section
#####################

def create_thread():
    """
    Create a new conversation thread for interacting with the assistant.
    """
    return client.beta.threads.create()

def add_message_to_thread(thread_id, content):
    """
    Add a user's message to the conversation thread.
    """
    return client.beta.threads.messages.create(thread_id=thread_id, role="user", content=content)

def run_assistant_on_thread(thread_id, assistant_id):
    """
    Trigger the assistant to process the messages in the thread.
    """
    return client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

def wait_for_run_completion(thread_id, run):
    """
    Poll the API until the assistant run is complete and print the responses.
    """
    while run.status not in ['completed', 'failed', 'cancelled', 'expired']:
        print("Waiting for Reese Istor to respond...")
        time.sleep(SLEEP_INTERVAL)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    if run.status == 'completed':
        print_assistant_responses(thread_id)
    else:
        print(f"The run did not complete successfully. Status: {run.status}")

def print_assistant_responses(thread_id):
    """
    Print the assistant's responses after the run is complete.
    """
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == "assistant":
            print(msg.content)

#####################
# Main Interaction Loop
#####################

def main():
    """
    Run the main interaction loop allowing a user to ask questions to the assistant.
    """
    print("Welcome to Reese Istor, your power electronics expert!")
    thread_id = thread.id  # Access the ID attribute of the thread object

    while True:
        user_input = input("Ask Reese Istor a question about power electronics (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Thank you for using Reese Istor. Goodbye!")
            break

        add_message_to_thread(thread_id, user_input)
        run = run_assistant_on_thread(thread_id, assistant_id)
        wait_for_run_completion(thread_id, run)

if __name__ == '__main__':
    main()