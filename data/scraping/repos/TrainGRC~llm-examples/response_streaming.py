#!/usr/bin/env python3
import subprocess
import sys
import time

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

def openai_chat_completion(messages, max_retries=3):
    retries = 0
    backoff_time = 1
    full_response_content = ""
    while retries <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )
            for chunk in response:
                if 'choices' in chunk and len(chunk.choices) > 0:
                    if 'delta' in chunk.choices[0] and 'content' in chunk.choices[0]['delta']:
                        chunk_content = chunk.choices[0]['delta']['content']
                        print(colored(chunk_content, 'green'), end='', flush=True)  # Print each chunk to console as it comes in
                        full_response_content += chunk_content  # Store each chunk in a string
            return full_response_content  # Return the full response content at the end
        except openai.error as e:
            raise e

def truncate_by_removing(conversation_history):
    """Removes the oldest messages until the conversation history is short enough."""
    print(colored("Removing oldest messages.", 'red'))
    while num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        conversation_history.pop(0)
    return conversation_history

def truncate_by_summarizing(conversation_history):
    """Summarizes the oldest messages until the conversation history is short enough."""
    print(colored("Removing oldest messages.", 'red'))
    messages_to_remove = []
    while num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        messages_to_remove.append(conversation_history.pop(0))
    if messages_to_remove:
        messages_to_remove.append({"role": "system", "content": "You are responsible for summarizing the previous conversation."})
        summary = openai_chat_completion(messages_to_remove)
        print(colored(f"\nSummary of removed messages: \n{summary}", 'yellow'))
        conversation_history.insert(0, {"role": "assistant", "content": f'Summary of Removed Messages: {summary}'})
    return conversation_history

conversation_history = []
while True:
    if num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        conversation_history = truncate_by_summarizing(conversation_history)
    
    user_input = input("\nPlease enter your request (type 'q' or 'quit' to exit): ")
    if user_input.lower() in ['q', 'quit']:
        break

    new_message = {"role": "user", "content": user_input}
    conversation_history.append(new_message)

    print(colored(f"\n{num_tokens_from_messages(conversation_history)} tokens", 'green'))
    print(colored("\nFull conversation history:", 'blue'))
    for message in conversation_history:
        print(message)
    print("")
    response_content = openai_chat_completion(conversation_history)
    conversation_history.append({"role": "assistant", "content": response_content})
    print(f"\n")