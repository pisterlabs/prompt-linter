import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
         # In order not to extend the call in an emergency, please keep your replies on the briefer side, but still be polite to the caller. \
def run_conversation():

    messages = [
        {"role" : "system", "content": "You will be acting as a 911 dispatcher. \
         You will talk with the caller to extract important contextual information about the emergency. \
         Prioritize getting the location, any current injuries, and the current situation of the emergency immediately.\
         Only ask one question at a time. Ask your questions directly and precisely. Assume the dispatcher is local and knows existing street names and local locations."},
        {"role": "assistant", "content": "911, what's your emergency?"},
    ]

    print("911, what's your emergency?")
    messages.append({'role': 'user', 'content': input("User: ")})

    for i in range(5):

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        print(response["choices"][0]["message"]["content"])
        messages.append({'role': 'assistant', 'content': response["choices"][0]["message"]["content"]})

        new_message = {"role": "user", "content": input("User: ")}
        messages.append(new_message)

run_conversation()