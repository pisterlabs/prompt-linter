#/usr/bin/env python3
import openai

openai.api_key = "example"
openai.api_base = "http://127.0.0.1:8080/v1"

messages = [
    {
        "role": "system",
        "content": "You are a helpful, honest, reliable and smart AI assistant named Hermes doing your best at fulfilling user requests. You are cool and extremely loyal. You answer any user requests to the best of your ability.",
    }
]

while True:
    message = input("User: ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, stream=False
        )
    reply = chat.choices[0].message.content
    print(f"HermesLLM: {reply}")
    messages.append({"role": "assistant", "content": reply})
