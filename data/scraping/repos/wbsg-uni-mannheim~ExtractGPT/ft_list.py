import json
import random
from datetime import datetime
from typing import List
from collections import Counter

import click
import tiktoken
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pydantic import ValidationError, BaseModel
from tqdm import tqdm

from pieutils import create_pydanctic_models_from_known_attributes, save_populated_task
from pieutils.config import CHATGPT_FINETUNING
from pieutils.evaluation import evaluate_predictions
from pieutils.fusion import fuse_models
from pieutils.preprocessing import update_task_dict_from_test_set
from pieutils.search import load_train_for_vector_store

# We start by importing the required packages

import json
import os
import tiktoken
import numpy as np
from collections import defaultdict


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages. Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

@click.command()
@click.option('--dataset', default='mave', help='Dataset Name')
@click.option('--model', default='gpt-3.5-turbo-0613', help='Model name')
@click.option('--verbose', default=False, help='Verbose mode')
def main(dataset, model, verbose):
    # Load task template
    with open('prompts/task_template.json', 'r') as f:
        task_dict = json.load(f)

    task_dict['task_prefix'] = "Extract the attribute values from the product {part} below in a JSON format. Valid " \
                                "attributes are {attributes}. If an attribute is not present in the product title, " \
                                "the attribute value is supposed to be 'n/a'."

    #task_dict['data_source'] = dataset
    task_dict['dataset_name'] = dataset
    task_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load examples into task dict
    task_dict = update_task_dict_from_test_set(task_dict)
    train_records = load_train_for_vector_store(dataset, categories=None)

    pydantic_models = create_pydanctic_models_from_known_attributes(task_dict['known_attributes'])
    # Create chains to extract attribute values from product titles
    system_setting = SystemMessage(
        content='You are a world-class algorithm for extracting information in structured formats. ')
    human_task_prompt = HumanMessagePromptTemplate.from_template(task_dict['task_prefix'])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate(messages=[system_setting, human_task_prompt, human_message_prompt])

    formatted_training_records = []
    for record in train_records.values():
        formatted_prompt = prompt.format_prompt(input=record['input'], attributes=', '.join(task_dict['known_attributes'][record['category']]), part=record['part'] if record['part'] != 'input' else 'title')
        formatted_training_record = {'messages': []}
        formatted_message_prompts = formatted_prompt.to_messages()
        empty_message = False
        for i in range(len(formatted_message_prompts)):
            message = formatted_message_prompts[i]
            role = None
            if message.type == 'system':
                role = 'system'
            elif message.type == 'human':
                role = 'user'
            if len(message.content)>0:
                formatted_training_record['messages'].append({'role': role, 'content': message.content})
            else:
                empty_message = True
                print(f"Warning: empty message at index {i} in prompt {formatted_prompt}")

        if empty_message:
            continue

        pydantic_example = pydantic_models[record['category']](**record['extractions'])
        processed_message_content = pydantic_example.json(indent=4).replace('null', '"n/a"')
        formatted_training_record['messages'].append({'role': 'assistant', 'content': processed_message_content})

        formatted_training_records.append(formatted_training_record)

    # Count tokens
    num_tokens = 0
    for formatted_training_record in formatted_training_records:
        num_new_tokens = num_tokens_from_messages(formatted_training_record['messages'], model=model)
        if num_new_tokens > 4096:
            print(f"Warning: {num_new_tokens} tokens in a single training record. This exceeds the maximum of 4096 tokens per training record.")
        num_tokens += num_new_tokens
    print(f"Number of tokens: {num_tokens}")

    # # Split training records into train and validation set
    # random.shuffle(formatted_training_records)
    # train_size = int(len(formatted_training_records) * 0.9)
    # formatted_train_records = formatted_training_records[:train_size]
    # formatted_validation_records = formatted_training_records[train_size:]

    ## SAVE FORMATTED DATA
    directory_path_preprocessed_mave = f'{CHATGPT_FINETUNING}/{dataset}'
    if not os.path.exists(directory_path_preprocessed_mave):
        os.makedirs(directory_path_preprocessed_mave)

    with open(f'{directory_path_preprocessed_mave}/ft_multiple_attribute_values-{dataset}.jsonl', 'w') as f:
        json_messages = [json.dumps(messages) for messages in formatted_training_records]
        f.write('\n'.join(json_messages))

if __name__ == '__main__':
    load_dotenv()
    random.seed(42)
    main()
