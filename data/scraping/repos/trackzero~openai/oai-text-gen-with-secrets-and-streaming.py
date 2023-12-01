import os
import json
import tiktoken
import sys
from openai import OpenAI
from colorama import init, Fore, Style
import botocore
import botocore.session
from botocore.exceptions import ClientError
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
oaiclient=OpenAI()

model = "gpt-4" # "gpt-3.5-turbo"  # "gpt-4" if you have it.

# track total tokens used in session
session_tokens = 0
session_warning_delivered = False


# Retrieve API key from AWS Secrets Manager, or try environment variable
def get_secret():
    secret_name = "openai_api_key"
    region_name = "us-west-2"

    # Create boto client/session
    try:
        client = botocore.session.get_session().create_client('secretsmanager', region_name=region_name)
        cache_config = SecretCacheConfig()
        cache = SecretCache(config=cache_config, client=client)

        # Retrieve secret and cache
        secret = cache.get_secret_string(secret_name)
        secret_dict = json.loads(secret)

        return secret_dict['openai_api_key']
    except ClientError as e:
        # Print error if secret retrieval fails
        print(f"{Fore.YELLOW}Error retrieving secret: {e}{Style.RESET_ALL}")
        # Try environment variable instead
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key:
            print(f"{Fore.YELLOW}Trying environment variable instead of AWS Secrets Manager{Style.RESET_ALL}")
            return openai_api_key
        else:
            print(f"{Fore.RED}Environment variable OPENAI_API_KEY not found{Style.RESET_ALL}")


# Set up initial conversation context
conversation = []

# Set up colorama
init()

# from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model=model):
    """Returns the number of tokens used by a list of messages."""
    global session_warning_delivered
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model not found. Using cl100k_base encoding." if not session_warning_delivered else "")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print(f"Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301." if not session_warning_delivered else "")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(f"Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314." if not session_warning_delivered else "")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for item in conversation:
            if isinstance(item, dict):
                for key, value in item.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            elif isinstance(item, str):
                num_tokens += len(encoding.encode(item))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    session_warning_delivered = True
    return num_tokens


# Create an instance of the ChatCompletion API
def chatbot(conversation):
    max_tokens = 4096
    try:
        response = oaiclient.chat.completions.create(
            model=model,
            messages=conversation,
            max_tokens=max_tokens,
            stream=True)

        # event variables
        collected_chunks = []
        collected_messages = ""

        # capture and print event stream
        print(f"{Fore.YELLOW}{Style.BRIGHT}Bot: {Style.RESET_ALL}")
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk.choices[0].delta  # extract the message
            
            if chunk_message.content is not None:
                message_text = chunk_message.content
                collected_messages += message_text
                print(f"{message_text}", end="")
        print(f"\n")
        return collected_messages
    except Exception as e:
        # Print error if chatbot fails to generate response
        print(f"{Fore.RED}{Style.BRIGHT}Error generating chat response: {e}{Style.RESET_ALL}")


# Print welcome message and instructions
print(f"{Fore.GREEN}{Style.BRIGHT}Welcome to the chatbot! To start, enter your message below.")
print("To reset the conversation, type 'reset' or 'let's start over'.")
print("To stop, say 'stop','exit', or 'bye'" + Style.RESET_ALL)

# Loop to continuously prompt user for input and get response from OpenAI
while True:
    try:
        user_input = input(f"{Fore.CYAN}{Style.BRIGHT}User: {Style.RESET_ALL}")
        if user_input.lower() in ["reset", "let's start over"]:
            conversation = []
            print(f"{Fore.YELLOW}{Style.BRIGHT}Bot: Okay, let's start over.{Style.RESET_ALL}")

        elif user_input.lower() in ["stop", "exit", "bye", "quit", "goodbye"]:
            print(f"{Fore.RED}{Style.BRIGHT}Bot: Okay, goodbye!\n  Session Tokens Used: {session_tokens})\n{Style.RESET_ALL}")
            break
        else:
            # Append user message to conversation context
            conversation.append({"role": "user", "content": user_input})
            # Generate chat completion
            chat = chatbot(conversation)

            # estimate tokens, add to running total.
            instance_tokens = num_tokens_from_messages(conversation, model)
            print(f"{Fore.MAGENTA}{Style.BRIGHT} Transaction tokens: {instance_tokens}\n{Style.RESET_ALL}")
            session_tokens += instance_tokens

            # Append bot message to conversation context
            conversation.append({"role": "assistant", "content": chat})
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        print(f"{Fore.RED}{Style.BRIGHT}Bot: Okay, goodbye!\n  Session Tokens Used: {session_tokens}\n{Style.RESET_ALL}")
        break
    except Exception as e:
        # Print error if an unexpected exception occurs
        print(f"{Fore.RED}{Style.BRIGHT}Unexpected error: {e}\n  Session Tokens Used: {session_tokens}\n{Style.RESET_ALL}")