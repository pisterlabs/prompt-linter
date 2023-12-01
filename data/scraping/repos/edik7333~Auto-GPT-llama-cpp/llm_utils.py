import time
import openai
from colorama import Fore
from llama_cpp import Llama
from api_conn import SendMessages
from config import Config

cfg = Config()


openai.api_key = cfg.openai_api_key


# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=cfg.temperature, max_tokens=None)->str:
    resContent = ""
    if cfg.llama_mode:
        """Create a chat completion locally"""
        llm = Llama(model_path=cfg.model_path, n_ctx=2056)
        if(temperature==None):
            temperature=0.8
        response = llm.create_chat_completion(messages, temperature=temperature)
        print(response)
        resContent = response["choices"][0]["message"]["content"]

    elif cfg.API_mode:
        resContent = SendMessages(messages)
    else:
        """Create a chat completion using the OpenAI API"""
        if cfg.use_azure:
            response = openai.ChatCompletion.create(
                deployment_id=cfg.openai_deployment_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        resContent = response.choices[0].message["content"]

    return resContent
