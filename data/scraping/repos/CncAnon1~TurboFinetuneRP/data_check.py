import json
import os
import glob
from collections import defaultdict

import tiktoken
from colorama import Fore, Back, Style

import openai

from modules import config

def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 3
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += 1
    num_tokens += 3
    return num_tokens

def check_moderation(messages, json_file):
    batch_size = 32

    for batch_start in range(0, len(messages), batch_size):
        batch = messages[batch_start:batch_start + batch_size]
        contents = [message["content"] for message in batch]
        
        mod_check = openai.Moderation.create(contents)["results"]

        for i, result in enumerate(mod_check):
            if result["flagged"]:
                global_index = batch_start + i
                print(f"{Fore.RED}Message {global_index} in file {json_file} got flagged!{Style.RESET_ALL}")
                print(f"Message contents (truncated): {messages[global_index]['content'][:50]}...")
                for name, flagged in result["categories"].items():
                    if flagged:
                        print(f"{Fore.RED}Flagged category {name} with the score {result['category_scores'][name]}{Style.RESET_ALL}")

def check_format(messages, filename):
    format_errors = defaultdict(int)
    
    convo_len = num_tokens_from_messages(messages)
    if convo_len > 4096:
        format_errors["token_limit"] = 1
    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["missing_content"] += 1

    check_moderation(messages, filename)

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

    return format_errors

def check():
    data_dir = config.chats_folder

    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    total_tokens = 0
    for json_file in json_files:
        with open(json_file, 'r') as f:
            messages = json.load(f)
        total_tokens += num_tokens_from_messages(messages)
        format_errors = check_format(messages, json_file)

        if format_errors:
            print(f"Found errors in file {json_file}:")
            for k, v in format_errors.items():
                if k == "token_limit":
                    print("The chat is over the 4096 token limit, it will be truncated during fine-tuning")
                else:
                    print(f"{k}: {v}")
        else:
            print(f"No errors found in file {json_file}")
    token_cost_1k = 0.008 # gpt-3.5-turbo fine-tune cost for 1k tokens
    training_cost = (total_tokens / 1000) * token_cost_1k * config.n_epochs
    print(f"Total token count is ~{total_tokens}, fine-tune will cost ~${training_cost:.2f} (epochs = {config.n_epochs})")