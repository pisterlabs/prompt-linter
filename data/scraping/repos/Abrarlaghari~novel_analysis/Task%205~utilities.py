#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def initialize_chat_model(api_key):
    chat_model = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        max_tokens=14000,
        openai_api_key=api_key,
        request_timeout=1500
    )
    return chat_model


def create_system_message_prompt(template):
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    return system_message_prompt


def create_human_message_prompt(template):
    human_message_prompt = HumanMessagePromptTemplate.from_template(template)
    return human_message_prompt


def create_chat_prompt(system_message_prompt, human_message_prompt):
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt


def extract_dialogues_from_chapter(chapter_file, chat_chain):
    chunk_size = 4500
    overlap = 500
    with open(chapter_file, 'r') as file:
        chapter_text = file.read()

    chunks = []
    start = 0
    end = chunk_size
    while start < len(chapter_text):
        chunks.append(chapter_text[start:end])
        start = end - overlap
        end = start + chunk_size

    dialogues = []
    for chunk in chunks:
        response = chat_chain.run(chnk=chunk)
        dialogues.append(response)

    return dialogues


def process_dialogues(dialogues):
    dialogue_collection_raw = []

    for diag in dialogues:
        tmp = diag.split("\n")
        for strngs in tmp:
            dialogue_collection_raw.append(strngs)

    return dialogue_collection_raw


def write_dialogues_to_file(dialogue_collection_raw, output_raw_file):
    with open(output_raw_file, 'w') as file:
        for line in dialogue_collection_raw:
            file.write("%s\n" % line)
            
            

if __name__ == "__main__":
    pass
