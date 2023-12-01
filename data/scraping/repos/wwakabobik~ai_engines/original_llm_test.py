# -*- coding: utf-8 -*-
"""
Filename: original_llm_test.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 21.11.2023
Last Modified: 21.11.2023

Description:
This file contains benchmarks for original LLMs models.
"""

import json

import asyncio
from cohere import Client as CohereClient
from llamaapi import LlamaAPI
from openai_python_api import ChatGPT

# pylint: disable=import-error
from examples.creds import oai_token, oai_organization, cohere_token, llama_token  # type: ignore
from examples.llm_api_comparison.csv_saver import save_to_csv
from examples.llm_api_comparison.llm_questions import llm_questions
from utils.llm_timer_wrapper import TimeMetricsWrapperAsync, TimeMetricsWrapperSync

# Initialize LLMs with tokens
llama = LlamaAPI(llama_token)
chatgpt_4 = ChatGPT(auth_token=oai_token, organization=oai_organization, stream=False)
chatgpt_3_5_turbo = ChatGPT(auth_token=oai_token, organization=oai_organization, stream=False, model="gpt-3.5-turbo")
cohere = CohereClient(cohere_token)


@TimeMetricsWrapperAsync
async def check_chat_gpt_4_response(prompt):
    """
    Check chat response from OpenAI API (ChatGPT-4).

    :param prompt: The prompt to use for the function.
    :type prompt: str
    """
    return await anext(chatgpt_4.str_chat(prompt=prompt))


@TimeMetricsWrapperAsync
async def check_chat_gpt_3_5_turbo_response(prompt):
    """
    Check chat response from OpenAI API (ChatGPT-3.5-Turbo).

    :param prompt: The prompt to use for the function.
    :type prompt: str
    """
    return await anext(chatgpt_3_5_turbo.str_chat(prompt=prompt))


@TimeMetricsWrapperSync
def check_chat_cohere_response(prompt):
    """
    Check chat response from Cohere.

    :param prompt: The prompt to use for the function.
    :type prompt: str
    """
    results = cohere.generate(prompt=prompt, max_tokens=100, stream=False)
    texts = [result.text for result in results][0]
    return texts


@TimeMetricsWrapperSync
def check_chat_llama_response(prompt):
    """
    Check chat response from Llama.

    :param prompt: The prompt to use for the function.
    :type prompt: str
    """
    # I won't implement wrapper for LLAMA here, but it's easy to do just reuse existing OpenAI wrapper.
    payload = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_length": 100,
        "temperature": 0.1,
        "top_p": 1.0,
        "frequency_penalty": 1.0,
    }
    response = llama.run(payload)
    response = json.dumps(response.json(), indent=2)
    response = json.loads(response)
    response = response["choices"][0]["message"]["content"]
    return response


# You can also add more public LLMs here, like:
# BardAI, https://www.bard.ai/ , you may try to use unofficial API: pip install bardapi
# Claude, https://claude.ai/ , you may try to use unofficial API: pip install claude-api


async def main():
    """Main function for benchmarking LLMs"""
    filename = "llms_orig.csv"
    for prompt in llm_questions:
        resp = await check_chat_gpt_4_response(prompt=prompt)
        save_to_csv(filename, "ChatGPT-4", prompt, resp)
        resp = await check_chat_gpt_3_5_turbo_response(prompt=prompt)
        save_to_csv(filename, "ChatGPT-3.5-Turbo", prompt, resp)
        resp = check_chat_cohere_response(prompt=prompt)
        save_to_csv(filename, "Cohere", prompt, resp)
        resp = check_chat_llama_response(prompt=prompt)
        save_to_csv(filename, "LLAMA", prompt, resp)


asyncio.run(main())
