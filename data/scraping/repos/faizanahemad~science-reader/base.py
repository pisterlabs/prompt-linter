from datetime import datetime
import sys
import random
from functools import partial
import glob
import traceback
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import inspect
import random
import inspect
from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholar import Paper
import mmh3
from pprint import pprint
import time
import concurrent.futures
import pandas as pd
import tiktoken
from copy import deepcopy, copy
import requests
import tempfile
from tqdm import tqdm
try:
    import ujson as json
except ImportError:
    import json
import requests
import dill
import os
from prompts import prompts
from langchain.document_loaders import MathpixPDFLoader
from functools import partial

from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI, ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from collections import defaultdict, Counter

import openai
import tiktoken
from vllm_client import get_streaming_vllm_response


from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    GPTVectorStoreIndex, 
    LangchainEmbedding, 
    LLMPredictor, 
    ServiceContext, 
    StorageContext, 
    download_loader,
    PromptHelper
)
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext

from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from llama_index.data_structs.node import Node, DocumentRelationship
from llama_index import LangchainEmbedding, ServiceContext, Document
from llama_index import GPTTreeIndex, SimpleDirectoryReader


from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import BingSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate

import tempfile
from flask_caching import Cache
temp_dir = tempfile.gettempdir()
import diskcache as dc
cache = dc.Cache(temp_dir)
cache_timeout = 7 * 24 * 60 * 60
# cache = Cache(None, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': temp_dir, 'CACHE_DEFAULT_TIMEOUT': 7 * 24 * 60 * 60})
try:
    from googleapiclient.discovery import build
except ImportError:
    raise ImportError(
        "google-api-python-client is not installed. "
        "Please install it with `pip install google-api-python-client`"
    )


import ai21

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', 100)

import logging
from common import *

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), "log.txt"))
    ]
)

from tenacity import (
    retry,
    RetryError,
    stop_after_attempt,
    wait_random_exponential,
)

import asyncio
import threading
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time


def get_embedding_model(keys):
    if "embeddingsUrl" in keys and not checkNoneOrEmpty(keys["embeddingsUrl"]):
        from embedding_client_server import EmbeddingClient
        return EmbeddingClient(keys["embeddingsUrl"])
    openai_key = keys["openAIKey"]
    assert openai_key
    # TODO: https://python.langchain.com/docs/modules/data_connection/caching_embeddings
    openai_embed = OpenAIEmbeddings(openai_api_key=openai_key, model='text-embedding-ada-002')
    return openai_embed

@retry(wait=wait_random_exponential(min=15, max=45), stop=stop_after_attempt(2))
def call_ai21(text, temperature=0.7, api_key=None):
    if get_gpt4_word_count(text) > 3600:
        logger.warning(f"call_ai21 Text too long, taking only first 3600 tokens")
        text = get_first_last_parts(text, 3600, 0)
    response_grande = ai21.Completion.execute(
          model="j2-jumbo-instruct",
          prompt=text,
          numResults=1,
          maxTokens=4000 - get_gpt4_word_count(text),
          temperature=temperature,
          topKReturn=0,
          topP=0.9,
          stopSequences=["##"],
          api_key=api_key["ai21Key"],
    )
    result = response_grande["completions"][0]["data"]["text"]
    return result
@retry(wait=wait_random_exponential(min=15, max=45), stop=stop_after_attempt(2))
def call_cohere(text, temperature=0.7, api_key=None):
    import cohere
    co = cohere.Client(api_key["cohereKey"])
    logger.debug(f"Calling Cohere with text: {text[:100]} and length: {len(text.split())}")
    if get_gpt4_word_count(text) > 3400:
        logger.warning(f"call_cohere Text too long, taking only first 3400 tokens")
        text = get_first_last_parts(text, 3400, 0)
    response = co.generate(
        model='command-nightly',
        prompt=text,
        max_tokens=3800 - get_gpt4_word_count(text),
        temperature=temperature)
    return response.generations[0].text


easy_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
davinci_enc = tiktoken.encoding_for_model("text-davinci-003")
gpt4_enc = tiktoken.encoding_for_model("gpt-4")
def call_chat_model(model, text, temperature, system, keys):
    api_key = keys["openAIKey"]
    response = openai.ChatCompletion.create(
        model=model,
        api_key=api_key,
        messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            stream=True
        )
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            yield chunk["choices"][0]["delta"]["content"]

    if chunk["choices"][0]["finish_reason"]!="stop":
        yield "\n Output truncated due to lack of context Length."


import requests
import json
import random

def fetch_completion_vllm(url, prompt, temperature, keys, max_tokens=4000, stream=False):
    # Define the headers for the request
    # append /v1/completions to the base URL if not already present in the URL
    if url.endswith("/generate"):
        url = url[:-len("/generate")]
    if url.endswith("/generate/"):
        url = url[:-len("/generate/")]

    prompt = get_first_last_parts(prompt, max_tokens - 1500, 1000, davinci_enc)
    input_len = len(davinci_enc.encode(prompt))
    assert max_tokens - input_len > 0
    max_tokens = max_tokens - input_len
    model = "Open-Orca/LlongOrca-13B-16k" # "Open-Orca/LlongOrca-13B-16k" # "lmsys/vicuna-13b-v1.5-16k"
    # Define the payload for the request
    if stream:
        if not url.endswith("/v1/") and not url.endswith("/v1"):
            url = url + "/v1"
        response = openai.ChatCompletion.create(
            model=model,
            api_key="EMPTY",
            api_base=url,
            messages=prompt,
            temperature=temperature,
            stream=True,
            stop_token_ids= [2],
            stop=["</s>", "Human:", "USER:", "[EOS]", "HUMAN:", "HUMAN :", "Human:", "User:", "USER :", "USER :",
                 "Human :", "###"],
        )
        yield "Response from a smaller 13B model.\n"
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]

        if chunk["choices"][0]["finish_reason"] != "stop":
            yield "\nOutput truncated due to lack of context Length."

    else:
        if not url.endswith("/v1/completions") and not url.endswith("/v1/completions/"):
            url = url + "/v1/completions"
        payload = {
            "model":model, # "LongLoRA/Llama-2-70b-chat-longlora-32k-sft", # "lmsys/vicuna-13b-v1.5-16k"
            'prompt': prompt,
            'max_tokens': min(max_tokens, 1024),
            'temperature': temperature,
            'stop_token_ids': [2],
            "stop": ["</s>", "Human:", "USER:", "[EOS]", "HUMAN:", "HUMAN :", "Human:", "User:", "USER :", "USER :", "Human :", "###"],
        }

        # Make the POST request
        response = requests.post(url, json=payload)

        # Parse the JSON response
        response_json = response.json()

        # Extract the 'finish_reason' and 'text' fields
        finish_reason = response_json['choices'][0]['finish_reason']
        text = response_json['choices'][0]['text']
        text = text.replace(prompt, "").strip()
        text = "Response from a smaller 13B model.\n" + text
        if finish_reason != 'stop':
            text += "\nOutput truncated due to lack of context length."

        yield text


def call_non_chat_model(model, text, temperature, system, keys):
    api_key = keys["openAIKey"]
    input_len = len(davinci_enc.encode(text))
    assert 4000 - input_len > 0
    completions = openai.Completion.create(
        api_key=api_key,
        engine=model,
        prompt=text,
        temperature=temperature,
        max_tokens = 4000 - input_len,
    )
    message = completions.choices[0].text
    finish_reason = completions.choices[0].finish_reason
    if finish_reason != 'stop':
        message = message + "\n Output truncated due to lack of context Length."
    return message

class CallLLmClaude:
    def __init__(self, keys, use_gpt4=False, use_16k=False, use_small_models=False, self_hosted_model_url=None):
        assert (use_gpt4 ^ use_16k ^ use_small_models) or (not use_gpt4 and not use_16k and not use_small_models)
        self.keys = keys
        self.self_hosted_model_url = self.keys["vllmUrl"] if not checkNoneOrEmpty(self.keys["vllmUrl"]) else None
        openai_basic_models = ["anthropic.claude-instant-v1"]
        openai_gpt4_models = ['anthropic.claude-v1', 'anthropic.claude-v2']
        openai_turbo_models = ['anthropic.claude-v1', 'anthropic.claude-v2']
        openai_16k_models = ['anthropic.claude-v1-100k', 'anthropic.claude-v2-100k']
        self.openai_basic_models = random.sample(openai_basic_models, len(openai_basic_models))
        self.openai_turbo_models = random.sample(openai_turbo_models, len(openai_turbo_models))
        self.openai_16k_models = random.sample(openai_16k_models, len(openai_16k_models))
        self.openai_gpt4_models = random.sample(openai_gpt4_models, len(openai_gpt4_models))
        use_gpt4 = use_gpt4 and self.keys.get("use_gpt4",
                                              True) and not use_small_models
        self.use_small_models = use_small_models
        self.use_gpt4 = use_gpt4 and len(openai_gpt4_models) > 0
        self.use_16k = use_16k and len(openai_16k_models) > 0

        self.gpt4_enc = tiktoken.encoding_for_model("gpt-4")
        self.turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.davinci_enc = tiktoken.encoding_for_model("text-davinci-003")
        self.system = "<role>You are a helpful research and reading assistant. Please follow the instructions and respond to the user request. Always provide detailed, comprehensive, thoughtful, insightful, informative and in-depth response. Directly start your answer without any greetings. End your response with ###. Write '###' after your response is over in a new line.</role>\n"
        import boto3
        self.bedrock = boto3.client(service_name='bedrock-runtime',
                               region_name=os.getenv("AWS_REGION"),
                               aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                               aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                               endpoint_url=os.getenv("BEDROCK_ENDPOINT_URL"))


    def call_claude(self, modelId, body):
        from botocore.exceptions import EventStreamError
        # raise EventStreamError(dict(), "test-error-for-checking-vllm-backup")
        response = self.bedrock.invoke_model_with_response_stream(body=body, modelId=modelId)
        if response.get('modelStreamErrorException') is not None or response.get('internalServerException') is not None:
            import boto3
            self.bedrock = boto3.client(service_name='bedrock-runtime',
                                        region_name=os.getenv("AWS_REGION"),
                                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                        endpoint_url=os.getenv("BEDROCK_ENDPOINT_URL"))
            response = self.bedrock.invoke_model_with_response_stream(body=body, modelId=modelId)
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    fragment = json.loads(chunk.get('bytes').decode())
                    text = fragment.get('completion')
                    stop_reason = fragment.get('stop_reason')
                    if stop_reason == 'max_tokens':
                        text = text + "\nOutput truncated due to lack of context Length."
                    yield text

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(4))
    def __call__(self, text, temperature=0.7, stream=False, max_tokens=None):
        from botocore.exceptions import EventStreamError
        text = f"{self.system}\n\n{text}\n"
        body = json.dumps({"prompt": f"\n\nHuman: {text}\nAssistant:", "max_tokens_to_sample": 2048 if max_tokens is None else max_tokens, "temperature": temperature, "stop_sequences":list(set(["\n\nHuman:", "###", "Human:", "human:", "HUMAN:"] + ["</s>", "Human:", "USER:", "[EOS]", "HUMAN:", "HUMAN :", "Human:", "User:", "USER :", "USER :", "Human :", "###"]))})
        accept = 'application/json'
        contentType = 'application/json'
        vllmUrl = self.self_hosted_model_url
        def vllmBackup(*args, **kwargs):
            return fetch_completion_vllm(vllmUrl, text, temperature, self.keys, max_tokens=12000, stream=stream)
        text_len = len(self.gpt4_enc.encode(text) if self.use_gpt4 else self.turbo_enc.encode(text))
        logger.info(
            f"CallLLM with temperature = {temperature}, stream = {stream} with text len = {len(text.split())}, token len = {text_len}")
        if self.use_gpt4:
            assert len(self.gpt4_enc.encode(text)) < 8000
            modelId = next(round_robin(self.openai_gpt4_models))
        elif self.use_16k and text_len > 3400:
            modelId = next(round_robin(self.openai_16k_models))
        elif self.use_small_models:
            assert len(self.gpt4_enc.encode(text)) < 4000
            modelId = next(round_robin(self.openai_basic_models))
        else:
            assert len(self.gpt4_enc.encode(text)) < 8000
            modelId = next(round_robin(self.openai_turbo_models))
        return call_with_stream(self.call_claude, stream, modelId, body, backup_function=vllmBackup if vllmUrl is not None else None)


class CallLLmGpt:
    def __init__(self, keys, use_gpt4=False, use_16k=False, use_small_models=False, self_hosted_model_url=None):
        
        assert (use_gpt4 ^ use_16k ^ use_small_models) or (not use_gpt4 and not use_16k and not use_small_models)
        self.keys = keys
        self.system = "You are a helpful assistant. Please follow the instructions and respond to the user request. Don't repeat what is told to you in the prompt. Always provide thoughtful, insightful, informative and in-depth response. Directly start your answer without any greetings.\n"
        available_openai_models = self.keys["openai_models_list"]
        self.self_hosted_model_url = self.keys["vllmUrl"] if not checkNoneOrEmpty(self.keys["vllmUrl"]) else None
        openai_gpt4_models = [] if available_openai_models is None else [m for m in available_openai_models if "gpt-4" in m and "-preview" not in m]
        use_gpt4 = use_gpt4 and self.keys.get("use_gpt4", True) and not use_small_models and self.self_hosted_model_url is None
        self.use_small_models = use_small_models
        self.use_gpt4 = use_gpt4 and len(openai_gpt4_models) > 0
        openai_turbo_models = ["gpt-3.5-turbo"] if available_openai_models is None else [m for m in available_openai_models if "gpt-3.5-turbo" in m and "16k" not in m and "instruct" not in m]
        openai_instruct_models = ["gpt-3.5-turbo-instruct"] if available_openai_models is None else [m for m in
                                                                                         available_openai_models if
                                                                                         "gpt-3.5-turbo-instruct" in m and "16k" not in m]
        openai_16k_models = ["gpt-3.5-turbo-16k"] if available_openai_models is None else [m for m in available_openai_models if "gpt-3.5-turbo-16k" in m]
        self.use_16k = use_16k and len(openai_16k_models) > 0
        openai_basic_models = [
            "text-davinci-003", "text-davinci-003", 
            "text-davinci-002",]
        
        self.openai_basic_models = random.sample(openai_basic_models, len(openai_basic_models))
        self.openai_turbo_models = random.sample(openai_turbo_models, len(openai_turbo_models))
        self.openai_16k_models = random.sample(openai_16k_models, len(openai_16k_models))
        self.openai_gpt4_models = random.sample(openai_gpt4_models, len(openai_gpt4_models))
        self.openai_instruct_models = random.sample(openai_instruct_models, len(openai_instruct_models))
        self.gpt4_enc = tiktoken.encoding_for_model("gpt-4")
        self.turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.davinci_enc = tiktoken.encoding_for_model("text-davinci-003")

        
    @retry(wait=wait_random_exponential(min=30, max=90), stop=stop_after_attempt(3))
    def __call__(self, text, temperature=0.7, stream=False, max_tokens=None):
        text_len = len(self.gpt4_enc.encode(text) if self.use_gpt4 else self.turbo_enc.encode(text))
        logger.debug(f"CallLLM with temperature = {temperature}, stream = {stream}, token len = {text_len}")
        if self.self_hosted_model_url is not None:
            return call_with_stream(fetch_completion_vllm, self.self_hosted_model_url, text, temperature, self.keys,
                                         max_tokens=max_tokens)
        else:
            assert self.keys["openAIKey"] is not None
            assert not self.use_small_models

        if self.use_gpt4 and len(self.openai_gpt4_models) > 0:
#             logger.info(f"Try GPT4 models with stream = {stream}, use_gpt4 = {self.use_gpt4}")
            try:
                assert text_len < 8000
            except AssertionError as e:
                text = get_first_last_parts(text, 4000, 3500, self.gpt4_enc)
            models = round_robin(self.openai_gpt4_models)
            try:
                model = next(models)
                return call_with_stream(call_chat_model, stream, model, text, temperature, self.system, self.keys)
            except Exception as e:
                if type(e).__name__ == 'AssertionError':
                    raise e
                if len(self.openai_gpt4_models) > 1:
                    model = next(models)
                elif len(self.openai_16k_models) > 0:
                    model = self.openai_16k_models[0]
                else:
                    raise e
                return call_with_stream(call_chat_model, stream, model, text, temperature, self.system, self.keys)
        elif not self.use_16k:
            models = round_robin(self.openai_turbo_models + self.openai_instruct_models)
            assert text_len < 3800
            try:
                model = next(models)
#                 logger.info(f"Try turbo model with stream = {stream}")
                return call_with_stream(call_chat_model if "instruct" not in model else call_non_chat_model, stream, model, text, temperature, self.system, self.keys)
            except Exception as e:
                if type(e).__name__ == 'AssertionError':
                    raise e
                if len(self.openai_turbo_models) > 1:
                    model = next(models)
                    fn = call_chat_model if "instruct" not in model else call_non_chat_model
                else:
                    models = round_robin(self.openai_instruct_models)
                    model = next(models)
                    fn = call_non_chat_model
                try:  
                    return call_with_stream(fn, stream, model, text, temperature, self.system, self.keys)
                except Exception as e:
                    if type(e).__name__ == 'AssertionError':
                        raise e
                    elif self.keys["ai21Key"] is not None:
                        return call_with_stream(call_ai21, stream, text, temperature, self.keys)
                    else:
                        raise e
        elif self.use_16k:
            if text_len > 3400:
                models = round_robin(self.openai_16k_models)
                logger.warning(f"Try 16k model with stream = {stream} with text len = {text_len}")
            else:
                models = round_robin(self.openai_turbo_models)
                logger.warning(f"Try Turbo model with stream = {stream} with text len = {text_len}")
            assert text_len < 15000
            try:
                model = next(models)
#                 logger.info(f"Try 16k model with stream = {stream}")
                return call_with_stream(call_chat_model if "instruct" not in model else call_non_chat_model, stream, model, text, temperature, self.system, self.keys)
            except Exception as e:
                if type(e).__name__ == 'AssertionError':
                    raise e
                if len(self.openai_16k_models) > 0 and text_len > 3400:
                    model = next(models)
                    fn = call_chat_model
                elif len(self.openai_turbo_models) > 0:
                    model = next(models)
                    fn = call_chat_model
                elif len(self.openai_instruct_models) > 0:
                    models = round_robin(self.openai_instruct_models)
                    model = next(models)
                    fn = call_non_chat_model
                else:
                    raise e
                try:
                    return call_with_stream(call_chat_model if "instruct" not in model else call_non_chat_model, stream, model, text, temperature, self.system, self.keys["openAIKey"])
                except Exception as e:
                    raise e
        else:
            raise ValueError("No model use criteria met")

CallLLm = CallLLmGpt if prompts.llm == "gpt4" else (CallLLmClaude if prompts.llm == "claude" else CallLLmGpt)
        
def split_text(text):
    # Split the text by spaces, newlines, and HTML tags
    chunks = re.split(r'( |\n|<[^>]+>)', text)
    
    # Find the middle index
    middle = len(chunks) // 2

    # Split the chunks into two halves
    first_half = ''.join(chunks[:min(middle+100, len(chunks)-1)])
    second_half = ''.join(chunks[max(0, middle-100):])
    
    yield first_half
    yield second_half
    
    




@AddAttribute('name', "MathTool")
@AddAttribute('description', """
MathTool:
    This tool takes a numeric expression as a string and provides the output for it.

    Input params/args: 
        num_expr (str): numeric expression to evaluate

    Returns: 
        str: evaluated expression answer

    Usage:
        `answer=MathTool(num_expr="2*3") # Expected answer = 6, # This tool needs no initialization`

    """)
def MathTool(num_expr: str):
    math_tool = load_tools(["llm-math"], llm=llm)[0]
    return math_tool._run(num_expr).replace("Answer: ", "")


@AddAttribute('name', "WikipediaTool")
@AddAttribute('description', """
WikipediaTool:
    This tool takes a phrase or key words and searches them over wikipedia, returns results from wikipedia as a str.

    Input params/args: 
        search_phrase (str): phrase to search over on wikipedia

    Returns: 
        str: searched paragraph on basis of search_phrase from wikipedia

    Usage:
        `answer=WikipediaTool(search_phrase="phrase to search") # This tool needs no initialization`

    """)
def WikipediaTool(search_phrase: str):
    tool = load_tools(["wikipedia"], llm=llm)[0]
    return tool._run(search_phrase)

enc = tiktoken.encoding_for_model("gpt-4")

@AddAttribute('name', "TextLengthCheck")
@AddAttribute('description', """
TextLengthCheck:
    Checks if the token count of the given `text_document` is smaller or lesser than the `threshold`.

    Input params/args: 
        text_document (str): document to verify if its length or word count or token count is less than threshold.
        threshold (int): Token count, text_document token count is below this then returns True

    Returns: 
        bool: whether length or token count is less than given threshold.

    Usage:
        `length_valid = TextLengthCheck(text_document="document to check length") # This tool needs no initialization`
        `less_than_ten = TextLengthCheck(text_document="document to check length", threshold=10)`

    """)
def TextLengthCheck(text_document: str, threshold: int=3400):
    assert isinstance(text_document, str)
    return len(enc.encode(text_document)) < threshold

@AddAttribute('name', "Search")
@AddAttribute('description', """
Search:
    This tool takes a search phrase, performs search over a web search engine and returns a list of urls for the search.

    Input params/args: 
        search_phrase (str): phrase or keywords to search over the web/internet.
        top_n (int): Number of webpages or results to return from search. Default is 5.

    Returns: 
        List[str]: List of webpage urls for given search_phrase, List length same as top_n input parameter.

    Usage:
        `web_url_list = Search(search_phrase="phrase to search") # This tool needs no initialization`
        
    Alternative Usage:
        `web_url_list = Search(search_phrase="phrase to search", top_n=20) # Get a custom number of results

    """)
def Search(search_phrase: str, top_n: int=5):
    return [r["link"] for r in  BingSearchAPIWrapper().results(search_phrase, top_n)]

@AddAttribute('name', "ChunkText")
@AddAttribute('description', """
ChunkText:
    This tool takes a text document and chunks it into given chunk size lengths, then returns a list of strings as chunked sub-documents.

    Input params/args: 
        text_document (str): document to create chunks from.
        chunk_size (int): Size of each chunk. Default is 3400, smaller chunk sizes are needed if downstream systems throw length error or token limit exceeded errors.

    Returns: 
        List[str]: text_chunks

    Usage:
        `text_chunks = ChunkText(text_document="document to chunk") # This tool needs no initialization`
        
    Alternative Usage:
        `text_chunks = ChunkText(text_document="document to chunk", chunk_size=1800) # Smaller chunk size, more chunks, but avoid token limit exceeded or length errors.

    """)
def ChunkText(text_document: str, chunk_size: int=3400, chunk_overlap:int=100):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text_document)


class Summarizer:
    def __init__(self, keys, is_detailed=False):
        self.keys = keys
        self.is_detail = is_detailed
        self.name = "Summariser"
        self.description = """
Summarizer:
    This tool takes a text document and summarizes it into a shorter version while preserving the main points and context. Useful when the document is too long and needs to be shortened before further processing.

    Input params/args: 
        text_document (str): document to summarize.

    Returns: 
        str: summarized_document.

    Usage:
        `summary = Summarizer()(text_document="document to summarize") # Note: this tool needs to be initialized first.`
    """
        template = f""" 
Write a {"detailed and informational " if is_detailed else ""}summary of the document below while preserving the main points and context, do not miss any important details, do not remove mathematical details and references.
Document to summarize is given below.
'''{{document}}'''

Write {"detailed and informational " if is_detailed else ""}Summary below.
"""
        self.prompt = PromptTemplate(
            input_variables=["document"],
            template=template,
        )
    @timer
    def __call__(self, text_document):
        prompt = self.prompt.format(document=text_document)
        return CallLLm(self.keys, use_gpt4=False)(prompt, temperature=0.5)
    
class ReduceRepeatTool:
    def __init__(self, keys):
        self.keys = keys
        self.name = "ReduceRepeatTool"
        self.description = """       
ReduceRepeatTool:
    This tool takes a text document reduces repeated content in the document. Useful when document has a lot of repeated content or ideas which can be mentioned in a shorter version.

    Input params/args: 
        text_document (str): document to summarize.

    Returns: 
        str: non_repeat_document.

    Usage:
        `non_repeat_document = ReduceRepeatTool()(text_document="document to to reduce repeats") # Note: this tool needs to be initialized first.`
        
    """
        self.prompt = PromptTemplate(
            input_variables=["document"],
            template=""" 
Reduce repeated content in the document given. Remove redundant information. Some ideas or phrases or points are repeated with no variation, remove them, output non-repeated parts verbatim without any modification, do not miss any important details.
Document is given below.
'''{document}'''

Write reduced document after removing duplicate or redundant information below.
""",
        )
    def __call__(self, text_document):
        prompt = self.prompt.format(document=text_document)
        result = CallLLm(self.keys, use_gpt4=False)(prompt, temperature=0.4)
        logger.info(f"ReduceRepeatTool with input as \n {text_document} and output as \n {result}")
        return result

process_text_executor = ThreadPoolExecutor(max_workers=32)
def contrain_text_length_by_summary(text, keys, threshold=2000):
    summariser = Summarizer(keys)
    tlc = partial(TextLengthCheck, threshold=threshold)
    return text if tlc(text) else summariser(text)

def process_text(text, chunk_size, my_function, keys):
    # Split the text into chunks
    chunks = [c.strip() for c in list(ChunkText(text, chunk_size)) if len(c.strip()) > 0]
    if len(chunks) == 0:
        return 'No relevant information found.'
    if len(chunks) > 1:
        futures = [process_text_executor.submit(my_function, chunk) for chunk in chunks]
        # Get the results from the futures
        results = [future.result() for future in futures]
    else:
        results = [my_function(chunk) for chunk in chunks]

    threshold = 512*3
    tlc = partial(TextLengthCheck, threshold=threshold)
    
    while len(results) > 1:
        logger.warning(f"--- process_text --- Multiple chunks as result. Results len = {len(results)} and type of results =  {type(results[0])}")
        assert isinstance(results[0], str)
        results = [process_text_executor.submit(contrain_text_length_by_summary, r, keys, threshold) for r in results]
        results = [future.result() for future in results]
        results = combine_array_two_at_a_time(results, '\n\n')
    assert len(results) == 1
    results = results[0]
    # threshold = 384
    # tlc = partial(TextLengthCheck, threshold=threshold)
    # if not tlc(results):
    #     summariser = Summarizer(keys, is_detailed=True)
    #     logger.warning(f"--- process_text --- Calling Summarizer on single result with single result len = {len(results.split())}")
    #     results = summariser(results)
    #     logger.warning(
    #         f"--- process_text --- Called Summarizer and final result len = {len(results.split())}")
    
    # if not tlc(results):
    #     logger.warning("--- process_text --- Calling ReduceRepeatTool")
    #     results = ReduceRepeatTool(keys)(results)
    assert isinstance(results, str)
    return results

async def get_url_content(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        title = await page.title()
        page_content = await page.content()
        # TODO: get rendered body
        page_content = await page.evaluate("""
        (() => document.body.innerText)()
        """)
        await browser.close()
        return {"title": title, "page_content": page_content}


class Cleaner:
    def __init__(self, keys, prompt=None, context=None):
        self.keys=keys
        self.instruction = """
You will be given unclean text fragments from web scraping a url.
Your goal is to return cleaned text without html tags and other irrelevant content (including code exception stack traces). 
If you are given a user request, instruction or query, then use that as well in filtering the information and return information relevant to the user query or instruction.
just extract relevant information if user query is given (Try to answer mostly in bullet points in this case.) else return cleaned text..
No creativity needed here.
Some context about the source document and user query is provided next, use the user query if provided and give very concise succint response.
        """ if prompt is None else prompt
        self.clean_now_follows = "\nActual text to be cleaned follows: \n"
        self.prompt = (self.instruction + " " + (context if context is not None else "") + " " + self.clean_now_follows) if prompt is None else prompt
        
    def clean_one(self, string, model=None):
        return CallLLm(self.keys, use_gpt4=False)(self.prompt + string, temperature=0.2)

    
    def clean_one_with_exception(self, string):
        try:
            cleaned_text = self.clean_one(string)
            return cleaned_text
        except Exception as e:
            exp_str = str(e)
            too_long = "maximum context length" in exp_str and "your messages resulted in" in exp_str
            if too_long:
                return " ".join([self.clean_one_with_exception(st) for st in split_text(string)])
            raise e
                
    def __call__(self, string, chunk_size=3400):
        return process_text(string, chunk_size, self.clean_one_with_exception, self.keys)

class GetWebPage:
    
    def __init__(self, keys):
        self.keys = keys
        self.name = "GetWebPage"
        self.description = """
GetWebPage:
    This tool takes a url link to a webpage and returns cleaned text content of that Page. Useful if you want to visit a page and get it's content. Optionally it can also take a user context or instruction and give only relevant parts of the page for the provided context.

    Input params/args: 
        url (str): url of page to visit
        context (str): user query/instructions/context about what to look for in this webpage

    Returns: 
        str: page_content

    Usage:
        `page_content = GetWebPage()(url="url to visit", context="user query or page reading instructions") # Note: this tool needs to be initialized first.`

    """
    def __call__(self, url, context=None):
        page_items = run_async(get_url_content, url)

        if not isinstance(page_items, dict):
            print(f"url: {url}, title: None, content: None")
            return f"url: {url}, title: None, content: None"
        page_content = page_items["page_content"]
        if not isinstance(page_content, str):
            print(f"url: {url}, title: {page_items['title']}, content: None")
            return f"url: {url}, title: {page_items['title']}, content: None"
        page_content = Cleaner(self.keys, context=f"\n\n url: {url}, title: {page_items['title']}" + (f"user query or context: {context}" if context is not None else ""))(page_content,
        chunk_size=768)
        return f"url: {url}, title: {page_items['title']}, content: {page_content}"
    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.__call__(url)
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class ContextualReader:
    def __init__(self, keys, provide_short_responses=False):
        self.keys = keys
        self.name = "ContextualReader"
        self.provide_short_responses = provide_short_responses
        self.description = """
ContextualReader:
    This tool takes a context/query/instruction, and a text document. It reads the document based on the context or query instruction and outputs only parts of document relevant to the query. Useful when the document is too long and you need to store a short contextual version of it for answering the user request. Sometimes rephrasing the query/question/user request before asking the ContextualReader helps ContextualReader provide better results. You can also specify directives to ContextualReader like "return numbers only", along with the query for better results.

    Input params/args: 
        context_user_query (str): instructions or query on how to read the document to provide contextually useful content from the document.
        text_document (str): document to read and provide information from using context_user_query.

    Returns: 
        str: contextual_content_from_document

    Usage:
        `contextual_content_from_document = ContextualReader()(context_user_query="instructions on how to read document", text_document="document to read") # Note: this tool needs to be initialized first.`

    """
        # Use markdown formatting to typeset or format your answer better.
        long_or_short = "Provide a short, concise and informative response in 3-4 sentences. \n" if provide_short_responses else "Always provide detailed, comprehensive, thoughtful, insightful, informative and in-depth response covering the entire details. \n"
        response_prompt = "Write short, concise and informative" if provide_short_responses else "Remember to write detailed, comprehensive, thoughtful, insightful, informative and in depth"
        self.prompt = PromptTemplate(
            input_variables=["context", "document"],
            template=f"""You are an AI expert in question answering. {long_or_short}
Provide relevant and helpful information from the given document for the given user question and conversation context given below.
'''{{context}}'''

Document to read and extract information from is given below.
'''
{{document}}
'''

Output any relevant equations if found in latex format.
{response_prompt} response below.
""",
        )
        
    def get_one(self, context, chunk_size, document,):
        import inspect
        prompt = self.prompt.format(context=context, document=document)
        callLLm = CallLLm(self.keys, use_gpt4=False, use_16k=chunk_size>(TOKEN_LIMIT_FOR_SHORT * 1.4), use_small_models=False)
        result = callLLm(prompt, temperature=0.4, stream=False)
        assert isinstance(result, str)
        return result
        
        
    
    def get_one_with_exception(self, context, chunk_size, document):
        try:
            text = self.get_one(context, chunk_size, document)
            return text
        except Exception as e:
            exp_str = str(e)
            too_long = "maximum context length" in exp_str and "your messages resulted in" in exp_str
            if too_long:
                logger.warning(f"ContextualReader:: Too long context, raised exception {str(e)}")
                return " ".join([self.get_one_with_exception(context, st) for st in split_text(document)])
            raise e

    def __call__(self, context_user_query, text_document, chunk_size=TOKEN_LIMIT_FOR_SHORT):
        assert isinstance(text_document, str)
        import functools
        part_fn = functools.partial(self.get_one_with_exception, context_user_query, chunk_size)
        result = process_text(text_document, chunk_size, part_fn, self.keys)
        short = self.provide_short_responses and chunk_size < int(TOKEN_LIMIT_FOR_SHORT*1.4)
        result = get_first_last_parts(result, 384, 256) if short else get_first_last_parts(result, 1024, 1024)
        assert isinstance(result, str)
        return result

@typed_memoize(cache, str, int, tuple, bool)
def call_contextual_reader(query, document, keys, provide_short_responses=False, chunk_size=TOKEN_LIMIT_FOR_SHORT//2)->str:
    assert isinstance(document, str)
    cr = ContextualReader(keys, provide_short_responses=provide_short_responses)
    return cr(query, document, chunk_size=chunk_size+512)


import json
import re

def get_citation_count(dres):
    # Convert the dictionary to a JSON string and lowercase it
    json_string = json.dumps(dres).lower()
    
    # Use regex to search for the citation count
    match = re.search(r'cited by (\d+)', json_string)
    
    # If a match is found, return the citation count as an integer
    if match:
        return int(match.group(1))
    
    # If no match is found, return zero
    return ""

def get_year(dres):
    # Check if 'rich_snippet' and 'top' exist in the dictionary
    if 'rich_snippet' in dres and 'top' in dres['rich_snippet']:
        # Iterate through the extensions
        for extension in dres['rich_snippet']['top'].get('extensions', []):
            # Use regex to search for the year
            match = re.search(r'(\d{4})', extension)

            # If a match is found, return the year as an integer
            if match:
                return int(match.group(1))

    # If no match is found, return None
    return None

def search_post_processing(query, results, only_science_sites=False, only_pdf=False):
    seen_titles = set()
    seen_links = set()
    dedup_results = []
    for r in results:
        title = r.get("title", "").lower()
        link = r.get("link", "").lower().replace(".pdf", '').replace("v1", '').replace("v2", '').replace("v3",
                                                                                                         '').replace(
            "v4", '').replace("v5", '').replace("v6", '').replace("v7", '').replace("v8", '').replace("v9", '')
        if title in seen_titles or len(title) == 0 or link in seen_links:
            continue
        if only_science_sites is not None and only_science_sites and "arxiv.org" not in link and "openreview.net" not in link:
            continue
        if only_science_sites is not None and not only_science_sites and ("arxiv.org" in link or "openreview.net" in link):
            continue
        if only_pdf is not None and not only_pdf and "pdf" in link:
            continue

        try:
            r["citations"] = get_citation_count(r)
        except:
            try:
                r["citations"] = int(r.get("inline_links", {}).get("cited_by", {}).get("total", "-1"))
            except:
                r["citations"] = None
        try:
            r["year"] = get_year(r)
        except:
            try:
                r["year"] = re.search(r'(\d{4})', r.get("publication_info", {}).get("summary", ""))
            except:
                r["year"] = None
        r['query'] = query
        _ = r.pop("rich_snippet", None)
        dedup_results.append(r)
        seen_titles.add(title)
        seen_links.add(link)
    return dedup_results

@typed_memoize(cache, str, int, tuple, bool)
def bingapi(query, key, num, our_datetime=None, only_pdf=True, only_science_sites=True):
    from datetime import datetime, timedelta
    if our_datetime:
        now = datetime.strptime(our_datetime, "%Y-%m-%d")
        two_years_ago = now - timedelta(days=365*3)
        date_string = two_years_ago.strftime("%Y-%m-%d")
    else:
        now = None
    search = BingSearchAPIWrapper(bing_subscription_key=key, bing_search_url="https://api.bing.microsoft.com/v7.0/search")
    
    pre_query = query
    after_string = f"after:{date_string}" if now and not only_pdf and not only_science_sites else ""
    search_pdf = " filetype:pdf" if only_pdf else ""
    site_string = " (site:arxiv.org OR site:openreview.net) " if only_science_sites and not only_pdf else " "
    query = f"{query}{site_string}{after_string}{search_pdf}"
    results = search.results(query, num)
    dedup_results = search_post_processing(pre_query, results, only_science_sites=only_science_sites, only_pdf=only_pdf)
    logger.debug(f"Called BING API with args = {query}, {key}, {num}, {our_datetime}, {only_pdf}, {only_science_sites} and responses len = {len(dedup_results)}")
    
    return dedup_results

@typed_memoize(cache, str, int, tuple, bool)
def googleapi(query, key, num, our_datetime=None, only_pdf=True, only_science_sites=True):
    from langchain.utilities import GoogleSearchAPIWrapper
    from datetime import datetime, timedelta
    num=min(num, 20)
    
    if our_datetime:
        now = datetime.strptime(our_datetime, "%Y-%m-%d")
        two_years_ago = now - timedelta(days=365*3)
        date_string = two_years_ago.strftime("%Y-%m-%d")
    else:
        now = None
    cse_id = key["cx"]
    google_api_key = key["api_key"]
    service = build("customsearch", "v1", developerKey=google_api_key)

    search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=cse_id)
    pre_query = query
    after_string = f"after:{date_string}" if now else ""
    search_pdf = " filetype:pdf" if only_pdf else ""
    site_string = " (site:arxiv.org OR site:openreview.net) " if only_science_sites else " -site:arxiv.org AND -site:openreview.net "
    query = f"{query}{site_string}{after_string}{search_pdf}"
    
    results = search.results(query, min(num, 10), search_params={"filter":"1", "start": "1"})
    if num > 10:
        results.extend(search.results(query, min(num, 10), search_params={"filter":"1", "start": "11"}))
    dedup_results = search_post_processing(pre_query, results, only_science_sites=only_science_sites, only_pdf=only_pdf)
    logger.debug(f"Called GOOGLE API with args = {query}, {num}, {our_datetime}, {only_pdf}, {only_science_sites} and responses len = {len(dedup_results)}")
    
    return dedup_results

@typed_memoize(cache, str, int, tuple, bool)
def serpapi(query, key, num, our_datetime=None, only_pdf=True, only_science_sites=True):
    from datetime import datetime, timedelta
    import requests
    
    if our_datetime:
        now = datetime.strptime(our_datetime, "%Y-%m-%d")
        two_years_ago = now - timedelta(days=365*3)
        date_string = two_years_ago.strftime("%Y-%m-%d")
    else:
        now = None

    
    location = random.sample(["New Delhi", "New York", "London", "Berlin", "Sydney", "Tokyo", "Washington D.C.", "Seattle", "Amsterdam", "Paris"], 1)[0]
    gl = random.sample(["us", "uk", "fr", "ar", "ci", "dk", "ec", "gf", "hk", "is", "in", "id", "pe", "ph", "pt", "pl"], 1)[0]
    # format the date as YYYY-MM-DD
    
    url = "https://serpapi.com/search"
    pre_query = query
    after_string = f"after:{date_string}" if now else ""
    search_pdf = " filetype:pdf" if only_pdf else ""
    site_string = " (site:arxiv.org OR site:openreview.net) " if only_science_sites else " -site:arxiv.org AND -site:openreview.net "
    query = f"{query}{site_string}{after_string}{search_pdf}"
    params = {
       "q": query,
       "api_key": key,
       "num": num,
       "no_cache": False,
        "location": location,
        "gl": gl,
       }
    response = requests.get(url, params=params)
    rjs = response.json()
    if "organic_results" in rjs:
        results = rjs["organic_results"]
    else:
        return []
    keys = ['title', 'link', 'snippet', 'rich_snippet', 'source']
    results = [{k: r[k] for k in keys if k in r} for r in results]
    dedup_results = search_post_processing(pre_query, results, only_science_sites=only_science_sites, only_pdf=only_pdf)
    logger.debug(f"Called SERP API with args = {query}, {key}, {num}, {our_datetime}, {only_pdf}, {only_science_sites} and responses len = {len(dedup_results)}")
    
    return dedup_results


@typed_memoize(cache, str, int, tuple, bool)
def gscholarapi(query, key, num, our_datetime=None, only_pdf=True, only_science_sites=True):
    from datetime import datetime, timedelta
    import requests

    if our_datetime:
        now = datetime.strptime(our_datetime, "%Y-%m-%d")
        two_years_ago = now - timedelta(days=365 * 3)
        date_string = two_years_ago.strftime("%Y-%m-%d")
    else:
        now = None
    # format the date as YYYY-MM-DD

    url = "https://serpapi.com/search"
    pre_query = query
    search_pdf = " filetype:pdf" if only_pdf else ""
    site_string = ""
    query = f"{query}{search_pdf}"
    params = {
        "q": query,
        "api_key": key,
        "num": num,
        "engine": "google_scholar",
        "no_cache": False,
    }
    response = requests.get(url, params=params)
    rjs = response.json()
    if "organic_results" in rjs:
        results = rjs["organic_results"]
    else:
        return []
    keys = ['title', 'link', 'snippet', 'rich_snippet', 'source']
    results = [{k: r[k] for k in keys if k in r} for r in results]
    dedup_results = search_post_processing(pre_query, results, only_science_sites=only_science_sites, only_pdf=only_pdf)
    logger.debug(
        f"Called SERP Google Scholar API with args = {query}, {key}, {num}, {our_datetime}, {only_pdf}, {only_science_sites} and responses len = {len(dedup_results)}")
    return dedup_results
    
# TODO: Add caching
from web_scraping import web_scrape_page



def get_page_content(link, playwright_cdp_link=None, timeout=10):
    text = ''
    title = ''
    try:
        from playwright.sync_api import sync_playwright
        playwright_enabled = True
        with sync_playwright() as p:
            if playwright_cdp_link is not None and isinstance(playwright_cdp_link, str):
                try:
                    browser = p.chromium.connect_over_cdp(playwright_cdp_link)
                except Exception as e:
                    logger.error(f"Error connecting to cdp link {playwright_cdp_link} with error {e}")
                    browser = p.chromium.launch(headless=True, args=['--disable-web-security', "--disable-site-isolation-trials"])
            else:
                browser = p.chromium.launch(headless=True, args=['--disable-web-security', "--disable-site-isolation-trials"])
            page = browser.new_page(ignore_https_errors=True, java_script_enabled=True, bypass_csp=True)
            url = link
            page.goto(url)
            # example_page = browser.new_page(ignore_https_errors=True, java_script_enabled=True, bypass_csp=True)
            # example_page.goto("https://www.example.com/")
            
            try:
                page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability.js")
                # page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability-readerable.js")
                page.wait_for_selector('body', timeout=timeout * 1000)
                page.wait_for_function("() => typeof(Readability) !== 'undefined' && document.readyState === 'complete'", timeout=10000)
                while page.evaluate('document.readyState') != 'complete':
                    time.sleep(0.1)
                result = page.evaluate("""(function execute(){var article = new Readability(document).parse();return article})()""")
            except Exception as e:
                # TODO: use playwright response modify https://playwright.dev/python/docs/network#modify-responses instead of example.com
                logger.warning(f"Trying playwright for link {link} after playwright failed with exception = {str(e)}")
                # traceback.print_exc()
                # Instead of this we can also load the readability script directly onto the page by using its content rather than adding script tag
                page.wait_for_selector('body', timeout=timeout * 1000)
                while page.evaluate('document.readyState') != 'complete':
                    time.sleep(0.1)
                init_html = page.evaluate("""(function e(){return document.body.innerHTML})()""")
                init_title = page.evaluate("""(function e(){return document.title})()""")
                # page = example_page
                page.goto("https://www.example.com/")
                page.evaluate(f"""text=>document.body.innerHTML=text""", init_html)
                page.evaluate(f"""text=>document.title=text""", init_title)
                logger.debug(f"Loaded html and title into page with example.com as url")
                page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability.js")
                page.wait_for_function("() => typeof(Readability) !== 'undefined' && document.readyState === 'complete'", timeout=10000)
                # page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability-readerable.js")
                page.wait_for_selector('body', timeout=timeout*1000)
                while page.evaluate('document.readyState') != 'complete':
                    time.sleep(0.1)
                result = page.evaluate("""(function execute(){var article = new Readability(document).parse();return article})()""")
            title = normalize_whitespace(result['title'])
            text = normalize_whitespace(result['textContent'])
                
            try:
                browser.close()
            except:
                pass
    except Exception as e:
        # traceback.print_exc()
        try:
            logger.debug(f"Trying selenium for link {link} after playwright failed with exception = {str(e)})")
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.wait import WebDriverWait
            from selenium.webdriver.common.action_chains import ActionChains
            from selenium.webdriver.support import expected_conditions as EC
            options = webdriver.ChromeOptions()
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--headless')
            driver = webdriver.Chrome(options=options)
            driver.get(link)
            add_readability_to_selenium = '''
                    function myFunction() {
                        if (document.readyState === 'complete') {
                            var script = document.createElement('script');
                            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability.js';
                            document.head.appendChild(script);

                            // var script = document.createElement('script');
                            // script.src = 'https://cdnjs.cloudflare.com/ajax/libs/readability/0.4.4/Readability-readerable.js';
                            // document.head.appendChild(script);
                        } else {
                            setTimeout(myFunction, 1000);
                        }
                    }

                    myFunction();
                '''
            try:
                driver.execute_script(add_readability_to_selenium)
                while driver.execute_script('return document.readyState;') != 'complete':
                    time.sleep(0.1)
                def document_initialised(driver):
                    return driver.execute_script("""return typeof(Readability) !== 'undefined' && document.readyState === 'complete';""")
                WebDriverWait(driver, timeout=timeout).until(document_initialised)
                result = driver.execute_script("""var article = new Readability(document).parse();return article""")
            except Exception as e:
                traceback.print_exc()
                # Instead of this we can also load the readability script directly onto the page by using its content rather than adding script tag
                init_title = driver.execute_script("""return document.title;""")
                init_html = driver.execute_script("""return document.body.innerHTML;""")
                driver.get("https://www.example.com/")
                logger.debug(f"Loaded html and title into page with example.com as url")
                driver.execute_script("""document.body.innerHTML=arguments[0]""", init_html)
                driver.execute_script("""document.title=arguments[0]""", init_title)
                driver.execute_script(add_readability_to_selenium)
                while driver.execute_script('return document.readyState;') != 'complete':
                    time.sleep(0.1)
                def document_initialised(driver):
                    return driver.execute_script("""return typeof(Readability) !== 'undefined' && document.readyState === 'complete';""")
                WebDriverWait(driver, timeout=timeout).until(document_initialised)
                result = driver.execute_script("""var article = new Readability(document).parse();return article""")
                
            title = normalize_whitespace(result['title'])
            text = normalize_whitespace(result['textContent'])
            try:
                driver.close()
            except:
                pass
        except Exception as e:
            if 'driver' in locals():
                try:
                    driver.close()
                except:
                    pass
        finally:
            if 'driver' in locals():
                try:
                    driver.close()
                except:
                    pass
    finally:
        if "browser" in locals():
            try:
                browser.close()
            except:
                pass
    return {"text": text, "title": title}
@typed_memoize(cache, str, int, tuple, bool)
def freePDFReader(url, page_ranges=None):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    if page_ranges:
        start, end = page_ranges.split("-")
        start = int(start) - 1
        end = int(end) - 1
        " ".join([pages[i].page_content for i in range(start, end+1)])
    return " ".join([p.page_content for p in pages])

class CustomPDFLoader(MathpixPDFLoader):
    def __init__(self, file_path, processed_file_format: str = "mmd",
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        **kwargs):
        from langchain.utils import get_from_dict_or_env
        from pathlib import Path
        self.file_path = file_path
        self.web_path = None
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            r = requests.get(self.file_path, verify=False, headers=headers)

            if r.status_code != 200:
                raise ValueError(
                    "Check the url of your file; returned status code %s"
                    % r.status_code
                )

            self.web_path = self.file_path
            self.temp_dir = tempfile.TemporaryDirectory()
            temp_pdf = Path(self.temp_dir.name) / "tmp.pdf"
            with open(temp_pdf, mode="wb") as f:
                f.write(r.content)
            self.file_path = str(temp_pdf)
        self.mathpix_api_key = get_from_dict_or_env(
            kwargs, "mathpix_api_key", "MATHPIX_API_KEY"
        )
        self.mathpix_api_id = get_from_dict_or_env(
            kwargs, "mathpix_api_id", "MATHPIX_API_ID"
        )
        self.processed_file_format = processed_file_format
        self.max_wait_time_seconds = max_wait_time_seconds
        self.should_clean_pdf = should_clean_pdf
        
        self.options = {"rm_fonts": True, 
                   "enable_tables_fallback":True}
        if self.processed_file_format != "mmd":
            self.options["conversion_formats"] = {self.processed_file_format: True},
        if "page_ranges" in kwargs and kwargs["page_ranges"] is not None:
            self.options["page_ranges"] = kwargs["page_ranges"]
        
    @property
    def data(self) -> dict:
        if os.path.exists(self.file_path):
            options = dict(**self.options)
        else:
            options = dict(url=self.file_path, **self.options)
        return {"options_json": json.dumps(options)}
    def clean_pdf(self, contents: str) -> str:
        contents = "\n".join(
            [line for line in contents.split("\n") if not line.startswith("![]")]
        )
        # replace the "\" slash that Mathpix adds to escape $, %, (, etc.
        contents = (
            contents.replace(r"\$", "$")
            .replace(r"\%", "%")
            .replace(r"\(", "(")
            .replace(r"\)", ")")
        )
        return contents
    

class PDFReaderTool:
    def __init__(self, keys):
        self.mathpix_api_id=keys['mathpixId']
        self.mathpix_api_key=keys['mathpixKey']
    @typed_memoize(cache, str, int, tuple, bool)
    def __call__(self, url, page_ranges=None):
        if self.mathpix_api_id is not None and self.mathpix_api_key is not None:
            
            loader = CustomPDFLoader(url, should_clean_pdf=True,
                              mathpix_api_id=self.mathpix_api_id, 
                              mathpix_api_key=self.mathpix_api_key, 
                              processed_file_format="mmd", page_ranges=page_ranges)
            data = loader.load()
            return data[0].page_content
        else:
            return freePDFReader(url, page_ranges)

@typed_memoize(cache, str, int, tuple, bool)
def get_semantic_scholar_url_from_arxiv_url(arxiv_url):
    import requests
    arxiv_id = arxiv_url.split("/")[-1].split(".")[0]
    semantic_scholar_api_url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    response = requests.get(semantic_scholar_api_url)
    if response.status_code == 200:
        semantic_scholar_id = response.json()["paperId"]
        semantic_url = f"https://www.semanticscholar.org/paper/{semantic_scholar_id}"
        return semantic_url
    raise ValueError(f"Couldn't parse arxiv url {arxiv_url}")

@typed_memoize(cache, str, int, tuple, bool)
def get_paper_details_from_semantic_scholar(arxiv_url):
    print(f"get_paper_details_from_semantic_scholar with {arxiv_url}")
    arxiv_id = arxiv_url.split("/")[-1].replace(".pdf", '').strip()
    from semanticscholar import SemanticScholar
    sch = SemanticScholar()
    paper = sch.get_paper(f"ARXIV:{arxiv_id}")
    return paper



# TODO: Add caching
def web_search_part1(context, doc_source, doc_context, api_keys, year_month=None,
                     previous_answer=None, previous_search_results=None, extra_queries=None,
                     gscholar=False, provide_detailed_answers=False):

    st = time.time()
    if extra_queries is None:
        extra_queries = []
    num_res = 10
    n_query = "two" if previous_search_results or len(extra_queries) > 0 else "two"
    n_query_num = 2
    pqs = []
    if previous_search_results:
        for r in previous_search_results:
            pqs.append(r["query"])
    doc_context = f"You are also given the research document: '''{doc_context}'''" if len(doc_context) > 0 else ""
    previous_answer = f"We also have the answer we have given till now for this question as '''{previous_answer}''', write new web search queries that can help expand and follow up on this answer." if previous_answer and len(
            previous_answer.strip()) > 10 else ''
    pqs = f"We had previously generated the following web search queries in our previous search: '''{pqs}''', don't generate these queries or similar queries - '''{pqs}'''" if len(pqs)>0 else ''
    prompt = prompts.web_search_prompt.format(context=context, doc_context=doc_context, previous_answer=previous_answer, pqs=pqs, n_query=n_query)
    if len(extra_queries) < 1:
        # TODO: explore generating just one query for local LLM and doing that multiple times with high temperature.
        query_strings = CallLLm(api_keys, use_gpt4=False)(prompt, temperature=0.5, max_tokens=100)
        query_strings.split("###END###")[0].strip()
        logger.debug(f"Query string for {context} = {query_strings}") # prompt = \n```\n{prompt}\n```\n
        query_strings = [q.strip() for q in parse_array_string(query_strings.strip())[:n_query_num]]

        if len(query_strings) == 0:
            query_strings = CallLLm(api_keys, use_gpt4=False)(prompt, temperature=0.2, max_tokens=100)
            query_strings.split("###END###")[0].strip()
            query_strings = [q.strip() for q in parse_array_string(query_strings.strip())[:n_query_num]]
        if len(query_strings) <= 1:
            query_strings = query_strings + [context]
        query_strings = query_strings + extra_queries
    else:
        query_strings = extra_queries
    
    rerank_available = "cohereKey" in api_keys and api_keys["cohereKey"] is not None and len(api_keys["cohereKey"].strip()) > 0
    serp_available = "serpApiKey" in api_keys and api_keys["serpApiKey"] is not None and len(api_keys["serpApiKey"].strip()) > 0
    bing_available = "bingKey" in api_keys and api_keys["bingKey"] is not None and len(api_keys["bingKey"].strip()) > 0
    google_available = ("googleSearchApiKey" in api_keys and api_keys["googleSearchApiKey"] is not None and len(api_keys["googleSearchApiKey"].strip()) > 0) and ("googleSearchCxId" in api_keys and api_keys["googleSearchCxId"] is not None and len(api_keys["googleSearchCxId"].strip()) > 0)
    rerank_query = "\n".join(query_strings)
    if rerank_available:
        import cohere
        co = cohere.Client(api_keys["cohereKey"])
        num_res = 10
    if len(extra_queries) > 0:
        num_res = 10
    
    if year_month:
        year_month = datetime.strptime(year_month, "%Y-%m").strftime("%Y-%m-%d")
    
    if gscholar and serp_available:
        serps = [get_async_future(gscholarapi, query, api_keys["serpApiKey"], num_res, our_datetime=year_month, only_pdf=None, only_science_sites=None) for query in
                 query_strings]
        logger.debug(f"Using SERP for Google scholar search, serps len = {len(serps)}")
    elif google_available:
        num_res = 10
        serps = [get_async_future(googleapi, query, dict(cx=api_keys["googleSearchCxId"], api_key=api_keys["googleSearchApiKey"]), num_res, our_datetime=year_month, only_pdf=None, only_science_sites=None) for query in query_strings]
        logger.debug(f"Using GOOGLE for web search, serps len = {len(serps)}")
    elif serp_available:
        serps = [get_async_future(serpapi, query, api_keys["serpApiKey"], num_res, our_datetime=year_month, only_pdf=None, only_science_sites=None) for query in query_strings]
        logger.debug(f"Using SERP for web search, serps len = {len(serps)}")
    elif bing_available:
        serps = [get_async_future(bingapi, query, api_keys["bingKey"], num_res, our_datetime=None, only_pdf=None, only_science_sites=None) for query in query_strings]
        logger.debug(f"Using BING for web search, serps len = {len(serps)}")
    else:
        serps = []
        logger.warning(f"Neither GOOGLE, Bing nor SERP keys are given but Search option choosen.")
        return {"text":'', "search_results": [], "queries": query_strings + ["Search Failed --- No API Keys worked"]}
    try:
        assert len(serps) > 0
        serps = [s.result() for s in serps]
        assert len(serps[0]) > 0
    except Exception as e:
        logger.error(f"Error in getting results from web search engines, error = {e}")
        if serp_available:
            serps = [get_async_future(serpapi, query, api_keys["serpApiKey"], num_res, our_datetime=year_month, only_pdf=None, only_science_sites=None) for query in query_strings]
            logger.debug(f"Using SERP for web search, serps len = {len(serps)}")
        elif bing_available:
            serps = [get_async_future(bingapi, query, api_keys["bingKey"], num_res, our_datetime=None, only_pdf=None, only_science_sites=None) for query in query_strings]
            logger.debug(f"Using BING for web search, serps len = {len(serps)}")
        else:
            return {"text":'', "search_results": [], "queries": query_strings}
        serps = [s.result() for s in serps]
    
    qres = [r for serp in serps for r in serp if r["link"] not in doc_source and doc_source not in r["link"]]
    logger.debug(f"Using Engine for web search, serps len = {len([r for s in serps for r in s])} Qres len = {len(qres)}")
    dedup_results = []
    seen_titles = set()
    seen_links = set()
    link_counter = Counter()
    title_counter = Counter()
    if previous_search_results:
        for r in previous_search_results:
            seen_links.add(r['link'])
    len_before_dedup = len(qres)
    for r in qres:
        title = r.get("title", "").lower()
        link = r.get("link", "").lower().replace(".pdf", '').replace("v1", '').replace("v2", '').replace("v3", '').replace("v4", '').replace("v5", '').replace("v6", '').replace("v7", '').replace("v8", '').replace("v9", '')
        link_counter.update([link])
        title_counter.update([link])
        if title in seen_titles or len(title) == 0 or link in seen_links or "youtube.com" in link or "twitter.com" in link:
            continue
        dedup_results.append(r)
        seen_titles.add(title)
        seen_links.add(link)
        
    len_after_dedup = len(dedup_results)
    logger.debug(f"Web search:: Before Dedup = {len_before_dedup}, After = {len_after_dedup}")
#     logger.info(f"Before Dedup = {len_before_dedup}, After = {len_after_dedup}, Link Counter = \n{link_counter}, title counter = \n{title_counter}")
        
    # Rerank here first

    # if rerank_available:
    #     st_rerank = time.time()
    #     docs = [r["title"] + " " + r.get("snippet", '') for r in dedup_results]
    #     rerank_results = co.rerank(query=rerank_query, documents=docs, top_n=8, model='rerank-english-v2.0')
    #     pre_rerank = dedup_results
    #     dedup_results = [dedup_results[r.index] for r in rerank_results]
    #     tt_rerank = time.time() - st_rerank
    #     logger.info(f"--- Cohere Reranked in {tt_rerank:.2f} --- rerank len = {len(dedup_results)}")
        # logger.info(f"--- Cohere Reranked in {tt_rerank:.2f} ---\nBefore Dedup len = {len_before_dedup}, rerank len = {len(dedup_results)},\nBefore Rerank = ```\n{pre_rerank}\n```, After Rerank = ```\n{dedup_results}\n```")
        
    # if rerank_available:
    #     pdfs = [pdf_process_executor.submit(get_pdf_text, doc["link"]) for doc in dedup_results]
    #     pdfs = [p.result() for p in pdfs]
    #     docs = [r["snippet"] + " " + p["small_text"] for p, r in zip(pdfs, dedup_results)]
    #     rerank_results = co.rerank(query=rerank_query, documents=docs, top_n=8, model='rerank-english-v2.0')
    #     dedup_results = [dedup_results[r.index] for r in rerank_results]
    #     pdfs = [pdfs[r.index] for r in rerank_results]
    #     logger.info(f"--- Cohere PDF Reranked --- rerank len = {len(dedup_results)}")
    #     logger.info(f"--- Cohere PDF Reranked ---\nBefore Dedup len = {len_before_dedup} \n rerank len = {len(dedup_results)}, After Rerank = ```\n{dedup_results}\n```")
    texts = None
    # if rerank_available:
    #     texts = [p["text"] for p in pdfs]
        
    # if rerank_available:
    #     for r in dedup_results_web:
    #         if "snippet" not in r:
    #             logger.warning(r)
    #     docs = [r["title"] + " " + r.get("snippet", '') for r in dedup_results_web]
    #     rerank_results = co.rerank(query=rerank_query, documents=docs, top_n=4, model='rerank-english-v2.0')
    #     pre_rerank = dedup_results_web
    #     dedup_results_web = [dedup_results_web[r.index] for r in rerank_results]

    dedup_results = list(round_robin_by_group(dedup_results, "query"))[:30]
    for r in dedup_results:
        cite_text = f"""{(f" Cited by {r['citations']}" ) if r['citations'] else ""}"""
        r["title"] = r["title"] + f" ({r['year'] if r['year'] else ''})" + f"{cite_text}"

    links = [r["link"] for r in dedup_results]
    titles = [r["title"] for r in dedup_results]
    contexts = [context +"? \n" + r["query"] for r in dedup_results] if len(dedup_results) > 0 else None
    all_results_doc = dedup_results
    web_links = None
    web_titles = None
    web_contexts = None
    variables = [all_results_doc, web_links, web_titles, web_contexts, api_keys, links, titles, contexts, api_keys, texts]
    variable_names = ["all_results_doc", "web_links", "web_titles", "web_contexts", "api_keys", "links", "titles", "contexts", "texts"]
    cut_off = 20 if provide_detailed_answers else 10
    for i, (var, name) in enumerate(zip(variables, variable_names)):
        if not isinstance(var, (list, str)):
            pass
        else:
            variables[i] = var[:cut_off]
    all_results_doc, web_links, web_titles, web_contexts, api_keys, links, titles, contexts, api_keys, texts = variables
    logger.info(f"Time taken to get web search links: {(time.time() - st):.2f}")
    return all_results_doc, links, titles, contexts, web_links, web_titles, web_contexts, texts, query_strings, rerank_query, rerank_available

def web_search(context, doc_source, doc_context, api_keys, year_month=None, previous_answer=None, previous_search_results=None, extra_queries=None, gscholar=False, provide_detailed_answers=False):
    part1_res = get_async_future(web_search_part1, context, doc_source, doc_context, api_keys, year_month, previous_answer, previous_search_results, extra_queries, gscholar, provide_detailed_answers)
    # all_results_doc, links, titles, contexts, web_links, web_titles, web_contexts, texts, query_strings, rerank_query, rerank_available = part1_res.result()
    part2_res = get_async_future(web_search_part2, part1_res, api_keys, provide_detailed_answers=provide_detailed_answers)
    return [wrap_in_future(get_part_1_results(part1_res)), part2_res] # get_async_future(get_part_1_results, part1_res)

def web_search_queue(context, doc_source, doc_context, api_keys, year_month=None, previous_answer=None, previous_search_results=None, extra_queries=None, gscholar=False, provide_detailed_answers=False):
    part1_res = get_async_future(web_search_part1, context, doc_source, doc_context, api_keys, year_month, previous_answer, previous_search_results, extra_queries, gscholar, provide_detailed_answers)
    # all_results_doc, links, titles, contexts, web_links, web_titles, web_contexts, texts, query_strings, rerank_query, rerank_available = part1_res.result()
    part2_res = web_search_part2_queue(part1_res, api_keys, provide_detailed_answers=provide_detailed_answers)
    return [wrap_in_future(get_part_1_results(part1_res)), part2_res] # get_async_future(get_part_1_results, part1_res)

def web_search_part2_queue(part1_res, api_keys, provide_detailed_answers=False):
    all_results_doc, links, titles, contexts, web_links, web_titles, web_contexts, texts, query_strings, rerank_query, rerank_available = part1_res.result()
    web_links = [] if web_links is None else web_links
    web_titles = [] if web_titles is None else web_titles
    web_contexts = [] if web_contexts is None else web_contexts
    links, titles, contexts, texts = links + web_links, titles + web_titles, contexts + web_contexts, (texts + ([''] * len(web_links))) if texts is not None else None
    read_queue = queued_read_over_multiple_links(links, titles, contexts, api_keys, texts, provide_detailed_answers=provide_detailed_answers)
    return read_queue

def get_part_1_results(part1_res):
    rs = part1_res.result()
    return {"search_results": rs[0], "queries": rs[8]}


def web_search_part2(part1_res, api_keys, provide_detailed_answers=False):
    result_queue = web_search_part2_queue(part1_res, api_keys, provide_detailed_answers=provide_detailed_answers)
    web_text_accumulator = []
    full_info = []
    qu_st = time.time()
    cut_off = 8 if provide_detailed_answers else 4
    while True:
        qu_wait = time.time()
        break_condition = len(web_text_accumulator) >= cut_off or (qu_wait - qu_st) > 30
        if break_condition and result_queue.empty():
            break
        one_web_result = result_queue.get()
        qu_et = time.time()
        if one_web_result is None:
            continue
        if one_web_result == FINISHED_TASK:
            break

        if one_web_result["text"] is not None and one_web_result["text"].strip() != "":
            web_text_accumulator.append(one_web_result["text"])
            logger.info(f"Time taken to get {len(web_text_accumulator)}-th web result: {(qu_et - qu_st):.2f}")
        if one_web_result["full_info"] is not None and isinstance(one_web_result["full_info"], dict):
            full_info.append(one_web_result["full_info"])
    web_text = "\n\n".join(web_text_accumulator)
    all_results_doc, links, titles, contexts, web_links, web_titles, web_contexts, texts, query_strings, rerank_query, rerank_available = part1_res.result()
    return {"text": web_text, "full_info": full_info, "search_results": all_results_doc, "queries": query_strings}


import multiprocessing
from multiprocessing import Pool

def process_link(link_title_context_apikeys, use_large_context=False):
    link, title, context, api_keys, text, detailed = link_title_context_apikeys
    key = f"process_link-{str([link, context, detailed, use_large_context])}"
    key = str(mmh3.hash(key, signed=False))
    result = cache.get(key)
    if result is not None and "full_text" in result and len(result["full_text"].strip()) > 0:
        return result
    st = time.time()
    link_data = download_link_data(link_title_context_apikeys)
    title = link_data["title"]
    text = link_data["full_text"]
    link_title_context_apikeys = (link, title, context, api_keys, text, detailed)
    summary = get_downloaded_data_summary(link_title_context_apikeys, use_large_context=use_large_context)["text"]
    logger.debug(f"Time for processing PDF/Link {link} = {(time.time() - st):.2f}")
    cache.set(key, {"link": link, "title": title, "text": summary, "exception": False, "full_text": text, "detailed": detailed},
              expire=cache_timeout)
    return {"link": link, "title": title, "text": summary, "exception": False, "full_text": text, "detailed": detailed}

from concurrent.futures import ThreadPoolExecutor

def download_link_data(link_title_context_apikeys):
    link, title, context, api_keys, text, detailed = link_title_context_apikeys
    key = f"download_link_data-{str([link])}"
    key = str(mmh3.hash(key, signed=False))
    result = cache.get(key)
    if result is not None and "full_text" in result and len(result["full_text"].strip()) > 0:
        result["full_text"] = result["full_text"].replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', '')
        return result
    link = convert_to_pdf_link_if_needed(link)
    is_pdf = is_pdf_link(link)
    link_title_context_apikeys = (link, title, context, api_keys, text, detailed)
    if is_pdf:
        result = read_pdf(link_title_context_apikeys)
    else:
        result = get_page_text(link_title_context_apikeys)
    if "full_text" in result and len(result["full_text"].strip()) > 0:
        result["full_text"] = result["full_text"].replace('<|endoftext|>', '\n').replace('endoftext',
                                                                                         'end_of_text').replace(
            '<|endoftext|>', '')
        cache.set(key, result, expire=cache_timeout)
    return result


def read_pdf(link_title_context_apikeys):
    link, title, context, api_keys, text, detailed = link_title_context_apikeys
    key = f"read_pdf-{str([link])}"
    key = str(mmh3.hash(key, signed=False))
    result = cache.get(key)
    if result is not None:
        return result
    st = time.time()
    # Reading PDF
    extracted_info = ''
    pdfReader = PDFReaderTool({"mathpixKey": None, "mathpixId": None})
    txt = text.replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', '')
    try:
        if len(text.strip()) == 0:
            txt = pdfReader(link).replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', '')
    except Exception as e:
        logger.error(f"Error reading PDF {link} with error {e}")
        txt = ''
        return {"link": link, "title": title, "context": context, "detailed":detailed, "exception": True, "full_text": txt}
    cache.set(key, {"link": link, "title": title, "context": context, "detailed": detailed, "exception": False, "full_text": txt},
              expire=cache_timeout)
    return {"link": link, "title": title, "context": context, "detailed":detailed, "exception": False, "full_text": txt}





def get_downloaded_data_summary(link_title_context_apikeys, use_large_context=False):
    link, title, context, api_keys, text, detailed = link_title_context_apikeys
    txt = text.replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', '')
    st = time.time()
    extracted_info = ''
    try:
        if len(text.strip()) > 0:
            chunked_text = ChunkText(
                txt, TOKEN_LIMIT_FOR_DETAILED if detailed else TOKEN_LIMIT_FOR_SHORT, 0)[0]
            logger.debug(f"Time for content extraction for link: {link} = {(time.time() - st):.2f}")
            extracted_info = call_contextual_reader(context, chunked_text,
                                                    api_keys, provide_short_responses=False,
                                                    chunk_size=((TOKEN_LIMIT_FOR_DETAILED + 500) if detailed else (TOKEN_LIMIT_FOR_SHORT + 200)))
            tt = time.time() - st
            logger.info(f"Called contextual reader for link: {link}, Result length = {len(extracted_info.split())} with total time = {tt:.2f}")
        else:
            chunked_text = text
            return {"link": link, "title": title, "text": extracted_info, "exception": True, "full_text": txt}
    except Exception as e:
        logger.error(f"Exception `{str(e)}` raised on `process_pdf` with link: {link}")
        return {"link": link, "title": title, "text": extracted_info, "detailed": detailed, "context": context, "exception": True, "full_text": txt}
    return {"link": link, "title": title, "context": context, "text": extracted_info, "detailed": detailed, "exception": False, "full_text": txt, "detailed": detailed}


def get_page_text(link_title_context_apikeys):
    link, title, context, api_keys, text, detailed = link_title_context_apikeys
    key = f"get_page_text-{str([link])}"
    key = str(mmh3.hash(key, signed=False))
    result = cache.get(key)
    if result is not None and "full_text" in result and len(result["full_text"].strip()) > 0:
        return result
    st = time.time()
    pgc = web_scrape_page(link, api_keys)
    if len(pgc["text"].strip()) == 0:
        logger.error(f"[process_page_link] Empty text for link: {link}")
        return {"link": link, "title": title, "exception": True, "full_text": '', "detailed": detailed, "context": context}
    title = pgc["title"]
    text = pgc["text"]
    cache.set(key, {"link": link, "title": title, "detailed": detailed, "exception": False, "full_text": text},
              expire=cache_timeout)
    return {"link": link, "title": title, "context": context, "exception": False, "full_text": text, "detailed": detailed}


pdf_process_executor = ThreadPoolExecutor(max_workers=32)

def queued_read_over_multiple_links(links, titles, contexts, api_keys, texts=None, provide_detailed_answers=False):
    basic_context = contexts[0] if len(contexts) > 0 else ""
    if texts is None:
        texts = [''] * len(links)

    link_title_context_apikeys = list(
        zip(links, titles, contexts, [api_keys] * len(links), texts, [provide_detailed_answers] * len(links)))
    link_title_context_apikeys = [[l] for l in link_title_context_apikeys]

    def call_back(result, *args, **kwargs):
        try:
            link = args[0][0][0]
        except:
            link = ''
        full_result = None
        text = ''

        if result is not None:
            assert isinstance(result, dict)
            result.pop("exception", None)
            result.pop("detailed", None)
            full_result = deepcopy(result)
            result.pop("full_text", None)
            text = f"[{result['title']}]({result['link']})\n{result['text']}"
        return {"text": text, "full_info": full_result, "link": link}

    threads = min(16 if provide_detailed_answers else 16, os.cpu_count()*8)
    # task_queue = orchestrator(process_link, list(zip(link_title_context_apikeys, [{}]*len(link_title_context_apikeys))), call_back, threads, 120)
    def fn1(link_title_context_apikeys, *args, **kwargs):
        link = link_title_context_apikeys[0]
        title = link_title_context_apikeys[1]
        context = link_title_context_apikeys[2]
        api_keys = link_title_context_apikeys[3]
        text = link_title_context_apikeys[4]
        detailed = link_title_context_apikeys[5]
        link_title_context_apikeys = (link, title, context, api_keys, text, detailed)
        return [download_link_data(link_title_context_apikeys), {}]
    def fn2(*args, **kwargs):
        link_data = args[0]
        link = link_data["link"]
        title = link_data["title"]
        text = link_data["full_text"]
        exception = link_data["exception"]
        context = link_data["context"] if "context" in link_data else basic_context
        detailed = link_data["detailed"] if "detailed" in link_data else False
        if exception:
            # link_data = {"link": link, "title": title, "text": '', "detailed": detailed, "context": context, "exception": True, "full_text": ''}
            raise Exception(f"Exception raised for link: {link}")
        link_title_context_apikeys = (link, title, context, api_keys, text, detailed)
        summary = get_downloaded_data_summary(link_title_context_apikeys)
        return summary
    def compute_timeout(link):
        return {"timeout": 60} if is_pdf_link(link) else {"timeout": 30}
    timeouts = list(pdf_process_executor.map(compute_timeout, links))
    # timeouts = [{"timeout": 30}] * len(links)
    task_queue = dual_orchestrator(fn1, fn2, list(zip(link_title_context_apikeys, timeouts)), call_back, threads, 30, 45)
    return task_queue

def read_over_multiple_links(links, titles, contexts, api_keys, texts=None, provide_detailed_answers=False):
    if texts is None:
        texts = [''] * len(links)
    # Combine links, titles, contexts and api_keys into tuples for processing
    link_title_context_apikeys = list(zip(links, titles, contexts, [api_keys] * len(links), texts, [provide_detailed_answers] * len(links)))
    # Use the executor to apply process_pdf to each tuple
    futures = [pdf_process_executor.submit(process_link, l_t_c_a, provide_detailed_answers and len(links) <= 2) for l_t_c_a in link_title_context_apikeys]
    # Collect the results as they become available
    processed_texts = [future.result() for future in futures]
    processed_texts = [p for p in processed_texts if not p["exception"]]
    # processed_texts = [p for p in processed_texts if not "no relevant information" in p["text"].lower()]
    # assert len(processed_texts) > 0
    if len(processed_texts) == 0:
        logger.warning(f"Number of processed texts: {len(processed_texts)}, with links: {links} in read_over_multiple_links")
    full_processed_texts = deepcopy(processed_texts)
    for p in processed_texts:
        p.pop("exception", None)
        p.pop("detailed", None)
        p.pop("full_text", None)
    # Concatenate all the texts

    # Cohere rerank here
    # result = "\n\n".join([json.dumps(p, indent=2) for p in processed_texts])
    if len(links) == 1:
        raw_texts = [ChunkText(p['full_text'].replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', ''),
                               TOKEN_LIMIT_FOR_SHORT - get_gpt4_word_count(p['text']) if provide_detailed_answers else TOKEN_LIMIT_FOR_SHORT//2 - get_gpt3_word_count(p['text']), 0)[0] for p in full_processed_texts]
        result = "\n\n".join([f"[{p['title']}]({p['link']})\nSummary:\n{p['text']}\nRaw article text:\n{r}\n" for r, p in
                              zip(raw_texts, processed_texts)])
    elif len(links) == 2 and provide_detailed_answers:
        raw_texts = [ChunkText(p['full_text'].replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', ''),
                               TOKEN_LIMIT_FOR_SHORT//2 - get_gpt4_word_count(
                                   p['text']) if provide_detailed_answers else TOKEN_LIMIT_FOR_SHORT//4 - get_gpt3_word_count(p['text']),
                               0)[0] for p in full_processed_texts]
        result = "\n\n".join([f"[{p['title']}]({p['link']})\nSummary:\n{p['text']}\nRaw article text:\n{r}\n" for r, p in
                              zip(raw_texts, processed_texts)])
    else:
        result = "\n\n".join([f"[{p['title']}]({p['link']})\n{p['text']}" for p in processed_texts])
    return result, full_processed_texts


def get_multiple_answers(query, additional_docs:list, current_doc_summary:str, provide_detailed_answers=False, provide_raw_text=True, dont_join_answers=False):
    # prompt = prompts.document_search_prompt.format(context=query, doc_context=current_doc_summary)
    # api_keys = additional_docs[0].get_api_keys()
    # query_strings = CallLLm(api_keys, use_gpt4=False)(prompt, temperature=0.5, max_tokens=100)
    # query_strings.split("###END###")[0].strip()
    # logger.debug(f"Query string for {query} = {query_strings}")  # prompt = \n```\n{prompt}\n```\n
    # query_strings = [q.strip() for q in parse_array_string(query_strings.strip())[:4]]
    # # select the longest string from the above array
    #
    # if len(query_strings) == 0:
    #     query_strings = CallLLm(api_keys, use_gpt4=False)(prompt, temperature=0.2, max_tokens=100)
    #     query_strings.split("###END###")[0].strip()
    #     query_strings = [q.strip() for q in parse_array_string(query_strings.strip())[:4]]
    # query_strings = sorted(query_strings, key=lambda x: len(x), reverse=True)
    # query_strings = query_strings[:1]
    # if len(query_strings) <= 0:
    #     query_strings = query_strings + [query]
    # query_string = query_strings[0]

    query_string = (f"Previous context and conversation details between human and AI assistant: '''{current_doc_summary}'''\n" if len(current_doc_summary.strip())>0 else '')+f"Provide {'detailed, comprehensive, thoughtful, insightful, informative and in depth' if provide_detailed_answers else ''} answer for this current query: '''{query}'''"

    futures = [pdf_process_executor.submit(doc.get_short_answer, query_string, defaultdict(lambda:False, {"scan": provide_detailed_answers}), True)  for doc in additional_docs]
    answers = [future.result() for future in futures]
    answers = [{"link": doc.doc_source, "title": doc.title, "text": answer} for answer, doc in zip(answers, additional_docs)]
    new_line = '\n\n'
    if len(additional_docs) == 1:
        if provide_raw_text:
            doc_search_results = [ChunkText(d.semantic_search_document(query), TOKEN_LIMIT_FOR_SHORT - get_gpt4_word_count(p['text']) if provide_detailed_answers else TOKEN_LIMIT_FOR_SHORT//2 - get_gpt3_word_count(p['text']), 0)[0] for p, d in zip(answers, additional_docs)]

            read_text = [f"[{p['title']}]({p['link']})\nAnswer:\n{p['text']}\nRaw article text:\n{r}\n" for r, p in
                                 zip(doc_search_results, answers)]
        else:
            read_text = [f"[{p['title']}]({p['link']})\nAnswer:\n{p['text']}" for p in
                 answers]
    elif len(additional_docs) == 2 and provide_detailed_answers:
        if False:
            doc_search_results = [ChunkText(d.semantic_search_document(query), 1400 - get_gpt4_word_count(p['text']) if provide_detailed_answers else 700 - get_gpt3_word_count(p['text']), 0)[0] for p, d in zip(answers, additional_docs)]
            read_text = [f"[{p['title']}]({p['link']})\nAnswer:\n{p['text']}\nRaw article text:\n{r}\n" for r, p in zip(doc_search_results, answers)]
        else:
            read_text = [f"[{p['title']}]({p['link']})\nAnswer:\n{p['text']}" for p in answers]
    else:
        read_text = [f"[{p['title']}]({p['link']})\n{p['text']}" for p in answers]
    if dont_join_answers:
        pass
    else:
        read_text = new_line.join(read_text)
    dedup_results = [{"link": doc.doc_source, "title": doc.title} for answer, doc in zip(answers, additional_docs)]
    logger.info(f"Query = ```{query}```\nAnswers = {answers}")
    return wrap_in_future({"search_results": dedup_results, "queries": [f"[{r['title']}]({r['link']})" for r in dedup_results]}), wrap_in_future({"text": read_text, "search_results": dedup_results, "queries": [f"[{r['title']}]({r['link']})" for r in dedup_results]})





