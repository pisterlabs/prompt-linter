# This code explores what we can do with the OpenAI Assistants API

# imports
from openai import OpenAI # ai stuff
from dotenv import load_dotenv
import os
import time # for delay

load_dotenv() # Load the environment
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

# https://platform.openai.com/docs/assistants/overview

# Assistant creation
director = client.beta.assistants.create(
    name="Movie Director: The Director of Strangeness",
    instructions="""
    Your name is the Director of Strangeness. You were born in a hole in the wall in Manhattan, year 2078. You are always completely confused. You have a personality that makes you create movies that are absolutely unhinged. You are a cultural critic, satirist, absurdist, and truly unpredictable movie director. Your works are considered the top of their class. Your grasp of juxtaposition, scale, atmosphere, and character dispositions are incredibly fine-grained and realistic.
    """,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

# A Thread represents a conversation. We recommend creating one Thread per user as soon 
# as the user initiates the conversation. Pass any user-specific context and files in 
# this thread by creating Messages.
thread = client.beta.threads.create()

# A Message contains the user's text
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content= """
    Write a script for a funny social simulation with 6 characters of the following trait:
    {
      "name": name of the character,
      "trait": their very comical trait,
      "plot": what they do in the story,
    }
    """
)

# This will run the assistant:
print("running the assistant...")
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=director.id,
  instructions="Please address the user as Jane Doe. The user has a premium account."
)

delay = 15

print(f"Waiting for {delay} seconds...")
time.sleep(delay)
print(f"{delay} seconds have passed!")

# Retrieve the messages added by the Assistant to the Thread
msg = client.beta.threads.messages.list(
  thread_id=thread.id
)

print(msg)

def parse_thread_messages(thread_messages):
    extracted_messages = []

    for thread_message in thread_messages:
        # Assuming each thread_message has a 'content' field which is a list
        for content in thread_message['content']:
            # Check if the content is of type 'text' and has a 'text' field
            if content['type'] == 'text' and 'text' in content:
                # Extracting the 'value' from the 'text' field
                text_content = content['text']['value']
                extracted_messages.append(text_content)

    return extracted_messages

# Assuming 'msg' is your provided object and it has a field 'data' which is a list of ThreadMessages
parsed_messages = parse_thread_messages(msg['data'])

# Printing the parsed messages
for message in parsed_messages:
    print(message)
    print("-" * 50)