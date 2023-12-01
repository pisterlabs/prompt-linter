# https://platform.openai.com/docs/api-reference/completions/create
import os
import openai

openai.api_key = "your api key"

messages = []
while True:
    newMsg = {"role": "user", "content": input("Me : ")}

    messages.append(newMsg)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    messages.append(completion.choices[0].message)
    print("Chatgpt :", completion.choices[0].message["content"])

    if newMsg["content"] == "bye":
        break
