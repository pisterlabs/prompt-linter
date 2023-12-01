import requests
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint
from time import sleep
import os
load_dotenv()

# Create an instance of OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def get_kanye_quote():
    r = requests.get("https://api.kanye.rest")
    if r.status_code == 200:
        return r.json()["quote"]
    else:
        return "Ran out of Kanye quotes, sorry"


# Create an assistant
# Speciazlies in fetching Kanye quotes
assistant = client.beta.assistants.create(
    name="Kanye Enthusiast",
    instructions="You are an advid follower of the teachings of Kanye West. Write and run code to get Kanye quotes.",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_kanye_quote",
            "description": "Returns a random Kanye West quote",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }],
    model="gpt-4-1106-preview"
)

# This is the channel that the user and assistant will communicate through
thread = client.beta.threads.create()

# User sends a message to the assistant
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need a Kanye quote in my life"
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

    if run.status == "requires_action":
        kanye_quote = get_kanye_quote()
        tools_to_call = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tool in tools_to_call:
            if tool.function.name == "get_kanye_quote":
                tool_outputs.append({ 
                     "tool_call_id": tool.id, 
                     "output": get_kanye_quote()
                 })

        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
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

