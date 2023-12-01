import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

"""
Refer
https://platform.openai.com/docs/api-reference/introduction
https://twitter.com/i/lists/1436436233509486592

"""

messages = []
system_msg = input("What type of chatbot would you like to create? \n")
messages.append({"role": "system", "content": system_msg})
print(messages)

print("Say hello to your new assistant!")
while input != "quit()":
    message = input("")
    messages.append({"role": "system", "content": message})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "system", "content": reply})
    print("\n" + reply + "\n")
