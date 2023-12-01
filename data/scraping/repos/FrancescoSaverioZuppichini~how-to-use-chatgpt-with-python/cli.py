import openai
import os
from rich.console import Console
from rich.markdown import Markdown

console = Console()

messages = [
    # system message first, it helps set the behavior of the assistant
    {"role": "system", "content": "You are a helpful assistant."},
]


def handle_keyboard_interrupt(n: int = 0):
    try:
        response = input("\nAre you sure you want to exit? (Y/N): ")
        if response.upper() == "Y":
            exit(0)
        else:
            return
    except KeyboardInterrupt:
        # we will allow two spammed ctrl+c, then bye bye
        if n < 1:
            handle_keyboard_interrupt(n + 1)
        return


try:
    while True:
        message = input("👨‍💻: ")
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            # doc is here https://platform.openai.com/docs/guides/chat/chat-vs-completions?utm_medium=email&_hsmi=248334739&utm_content=248334739&utm_source=hs_email
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        # get the reply
        reply = chat_completion.choices[0].message.content
        print("🤖: ", end="")
        console.print(Markdown(reply))
        messages.append({"role": "assistant", "content": reply})
except KeyboardInterrupt:
    handle_keyboard_interrupt()
