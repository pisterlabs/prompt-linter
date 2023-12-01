from __future__ import annotations

import json

from fastapi import WebSocket
import time

import openai
from langchain.adapters import openai as lc_openai
from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from agent.prompts import auto_agent_instructions
from config import Config
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
from langchain import PromptTemplate, LLMChain

CFG = Config()

openai.api_key = CFG.openai_api_key
openai.api_base = CFG.openai_api_base

from typing import Optional
import logging

def create_chat_completion(
    messages: list,  # type: ignore
    model: Optional[str] = None,
    temperature: float = CFG.temperature,
    max_tokens: Optional[int] = None,
    stream: Optional[bool] = False,
    websocket: WebSocket | None = None,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    Returns:
        str: The response from the chat completion
    """

    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 8001:
        raise ValueError(f"Max tokens cannot be more than 8001, but got {max_tokens}")
    if stream and websocket is None:
        raise ValueError("Websocket cannot be None when stream is True")

    # create response
    for attempt in range(10):  # maximum of 10 attempts
        response = send_chat_completion_request(
            messages, model, temperature, max_tokens, stream, websocket
        )
        return response

    logging.error("Failed to get response from OpenAI API")
    raise RuntimeError("Failed to get response from OpenAI API")

system = ""
llm = LlamaCpp(
            model_path="D:\GPT_Researcher\gpt-researcher\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
            temperature=0.75,
            max_tokens=16000,
            top_p=1,
            n_ctx= 4000,
            callback_manager=callback_manager,
            verbose=True,
        )
def send_chat_completion_request(
    messages, model, temperature, max_tokens, stream, websocket
):  
    if not stream:
        print(messages)
       
        template = """
        <s>[INST] <<SYS>>
          {system}
        <</SYS>>
        {user} [/INST]"""
   

        prompt = PromptTemplate(template=template, input_variables=["system","user"]) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        system = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
        for i in range(len(messages)):
            if messages[i]['role']=='system':
                system=messages[i]['content']
            if messages[i]['role']=='user':
                user=messages[i]['content'][5:]
        inputs = {"system": system, "user": user}
        response=llm_chain.run(inputs)
        return str(response)
    else:
        return stream_response(model, messages, temperature, max_tokens, websocket)


async def stream_response(model, messages, temperature, max_tokens, websocket):
    paragraph = ""
    response = ""
    print(f"streaming response...")

    template = """
    <s>[INST] <<SYS>>
        {system}
    <</SYS>>
    {user} [/INST]"""
    system='''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
    prompt = PromptTemplate(template=template, input_variables=["system","user"]) 
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    for i in range(len(messages)):
        if messages[i]['role']=='system':
            system=messages[i]['content']
        if messages[i]['role']=='user':
            user=messages[i]['content'][5:]
    inputs = {"system": system, "user": user}
    content=llm_chain.run(inputs)
    if content is not None:
            response += content
            paragraph += content
            if "\n" in paragraph:
                await websocket.send_json({"type": "report", "output": paragraph})
                paragraph = ""
    print(f"streaming response complete")
    return str(response)


def choose_agent(task: str) -> str:
    """Determines what agent should be used
    Args:
        task (str): The research question the user asked
    Returns:
        agent - The agent that will be used
        agent_role_prompt (str): The prompt for the agent
    """
    try:
        response = create_chat_completion(
            model=CFG.smart_llm_model,
             messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {task}"}],
            temperature=0,
        )
        response=response.lower()
        response=response.replace('"','')
        dict1={}
        try:
            dict1['agent']=response.split('agent:')[1].split('agent role prompt:')[0]
            dict1['agent_role_prompt']=response.split('agent:')[1].split('agent role prompt:')[1]
        except:
            dict1['agent']=response.split('agent:')[1].split('agent_role_prompt:')[0]
            dict1['agent_role_prompt']=response.split('agent:')[1].split('agent_role_prompt:')[1]
        dict1=json.dumps(dict1)
        return json.loads(dict1)
    except Exception as e:
        print(f"{Fore.RED}Error in choose_agent: {e}{Style.RESET_ALL}")
        print(1)
        return {"agent": "Default Agent",
                "agent_role_prompt": "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."}


