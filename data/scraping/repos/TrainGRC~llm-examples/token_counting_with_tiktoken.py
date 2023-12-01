#!/usr/bin/env python3
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import openai
except ImportError:
    print("Installing module: openai")
    install('openai')
    import openai

try:
    import tiktoken
except ImportError:
    print("Installing module: tiktoken")
    install('tiktoken')
    import tiktoken

try:
    from termcolor import colored
except ImportError:
    print("Installing module: termcolor")
    install('termcolor')
    from termcolor import colored

print("")
openai.api_key = input("Please enter your OpenAI API key: ")
MAX_TOKENS = 512

def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def openai_chat_completion(messages):
    """Returns the response from the OpenAI API given a list of messages."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message['content']

conversation_history = []
while True:
    user_input = input("\nPlease enter your request (type 'q' or 'quit' to exit): ")
    if user_input.lower() in ['q', 'quit']:
        break

    new_message = {"role": "user", "content": user_input}
    if num_tokens_from_messages(conversation_history + [new_message]) > MAX_TOKENS:
        print("Conversation history is too long and needs to be truncated.")
        continue

    conversation_history.append(new_message)
    print(colored("\nFull conversation history:", 'red'))
    for message in conversation_history:
        print(message)

    print(colored(f"\n{num_tokens_from_messages(conversation_history)} tokens", 'green'))

    response_content = openai_chat_completion(conversation_history)
    conversation_history.append({"role": "assistant", "content": response_content})
    print(f"\nResponse: \n{response_content}")