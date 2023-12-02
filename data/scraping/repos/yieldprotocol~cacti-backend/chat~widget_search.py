# This chat variant determines if the user's query is related to a widget or a search

import os
import time
from typing import Any, Callable

from langchain.prompts import PromptTemplate

import registry
import streaming
from .base import (
    BaseChat, ChatHistory, Response, ChatOutputParser,
)


WIDGET_INSTRUCTION = '''To help users, an assistant may display information or dialog boxes using magic commands. Magic commands have the structure "<|command(parameter1, parameter2, ...)|>". When the assistant uses a command, users will see data, an interaction box, or other inline item, not the command. Users cannot use magic commands. Fill in the command with parameters as inferred from the user input query. If there are missing parameters, prompt for them and do not make assumptions without the user's input. Do not return a magic command unless all parameters are known. Examples are given for illustration purposes, do not confuse them for the user's input. If a widget involves a transaction that requires user confirmation, prompt for it. If the widget requires a connected wallet, make sure that is available first. If there is no appropriate widget available, explain the situation and ask for more information. Do not make up a non-existent widget magic command, only use the most appropriate one. Here are the widgets that may match the user input:'''

SEARCH_INSTRUCTION = '''Information to help complete your task is below. Only use information below to answer the question, and create a final answer with inline citations linked to the provided source URLs. If you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer corresponding to the numbered inline citations.'''


TEMPLATE = '''You are a web3 assistant. You help users use web3 apps, such as Uniswap, AAVE, MakerDao, etc. You assist users in achieving their goals with these protocols, by providing users with relevant information, and creating transactions for users. Your responses should sound natural, helpful, cheerful, and engaging, and you should use easy to understand language with explanations for jargon.

{instruction}
---
{task_info}
---

User: {question}
Assistant:'''


# TODO: make this few-shot on real examples instead of dummy ones
IDENTIFY_TEMPLATE = '''
Given the following conversation and a follow up response input, determine if the input is related to a command to invoke using a widget or if it is a search query for a knowledge base. If it is a widget, return the appropriate keywords to search for the widget, as well as all relevant details to invoke it. If it is a search query, rephrase as a standalone question. You should assume that the query is related to web3.

## Example:

Chat History:
User: I'd like to make transfer ETH
Assistant: Ok I can help you with that. How much and to which address?

Input: 2 ETH to andy
Ouput: <widget> transfer of 2 ETH currency to andy

## Example:

Chat History:
User: Who created Ethereum?
Assistant: Vitalik Buterin
User: What about AAVE?
Assistant: Stani Kulechov

Input: When was that?
Output: <query> When were Ethereum and AAVE created?

## Example:

Chat History:
User: Who created Ethereum?
Assistant: Vitalik Buterin

Input: What is AAVE?
Output: <query> What is AAVE?

## Example:

Chat History:
User: What's my balance of USDC?
Assistant: Your USDC balance is <|balance('USDC')|>

Input: cost of ETH
Output: <widget> price of ETH coin given USDC balance

## Example:

Chat History:
{history}

Input: {question}
Output:'''


@registry.register_class
class WidgetSearchChat(BaseChat):
    def __init__(self, doc_index: Any, widget_index: Any, top_k: int = 3, show_thinking: bool = True) -> None:
        super().__init__()
        self.output_parser = ChatOutputParser()
        self.widget_prompt = PromptTemplate(
            input_variables=["task_info", "question"],
            template=TEMPLATE.replace("{instruction}", WIDGET_INSTRUCTION),
            output_parser=self.output_parser,
        )
        self.search_prompt = PromptTemplate(
            input_variables=["task_info", "question"],
            template=TEMPLATE.replace("{instruction}", SEARCH_INSTRUCTION),
            output_parser=self.output_parser,
        )
        self.doc_index = doc_index
        self.widget_index = widget_index
        self.top_k = top_k
        self.show_thinking = show_thinking

        self.identify_prompt = PromptTemplate(
            input_variables=["history", "question"],
            template=IDENTIFY_TEMPLATE,
            output_parser=self.output_parser,
        )

    def receive_input(self, history: ChatHistory, userinput: str, send: Callable) -> None:
        userinput = userinput.strip()
        # First identify the question
        history_string = history.to_string()
        start = time.time()
        example = {
            "history": history_string.strip(),
            "question": userinput,
            "stop": "##",
        }

        chat_message_id = None
        identify_response = ''
        identified_type = None
        sent_response = ''

        def identify_token_handler(token):
            nonlocal chat_message_id, identify_response, identified_type, sent_response
            if not self.show_thinking:
                return
            identify_response += token
            if '> ' not in identify_response.strip():
                return

            if not identified_type:
                identified_type, question = identify_response.strip().split(' ', 1)
                if identified_type == '<widget>':
                    token = "I think you want a widget for: " + question
                else:
                    token = "I think you're asking: " + question

            sent_response += token
            chat_message_id = send(Response(
                response=token,
                still_thinking=False,
                actor='bot',
                operation='append' if chat_message_id is not None else 'create',
            ), last_chat_message_id=chat_message_id)

        identify_chain = streaming.get_streaming_chain(self.identify_prompt, identify_token_handler)
        identify_response = identify_chain.apply_and_parse([example])[0]
        if self.show_thinking:
            # send again, but with replace so we save to db
            send(Response(
                response=sent_response,
                still_thinking=False,
                actor='bot',
                operation='replace',  # this will save to db where append did not
            ), last_chat_message_id=chat_message_id)

        duration = time.time() - start
        identified_type, question = identify_response.split(' ', 1)

        send(Response(
            response=f'Intent identification took {duration: .2f}s',
            actor='system',
            still_thinking=True,  # turn on thinking again
        ))

        chat_message_id = None

        def new_token_handler(token):
            nonlocal chat_message_id
            chat_message_id = send(Response(
                response=token,
                still_thinking=False,
                actor='bot',
                operation='append' if chat_message_id is not None else 'create',
            ), last_chat_message_id=chat_message_id)

        if identified_type == '<widget>':
            widgets = self.widget_index.similarity_search(question, k=self.top_k)
            task_info = '\n'.join([f'Widget: {widget.page_content}' for widget in widgets])
            chain = streaming.get_streaming_chain(self.widget_prompt, new_token_handler)
        else:
            docs = self.doc_index.similarity_search(question, k=self.top_k)
            task_info = '\n'.join([f'Content: {doc.page_content}\nSource: {doc.metadata["url"]}' for doc in docs])
            chain = streaming.get_streaming_chain(self.search_prompt, new_token_handler)
        start = time.time()
        example = {
            "task_info": task_info,
            "question": question,
            "stop": "User",
        }

        result = chain.apply_and_parse([example])[0]
        duration = time.time() - start
        history.add_interaction(userinput, result)
        send(Response(result, operation='replace'), last_chat_message_id=chat_message_id)
        send(Response(response=f'Response generation took {duration: .2f}s', actor='system'))
