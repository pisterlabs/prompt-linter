"""Retrieves model from OpenAI API and returns a ChatOpenAI object."""
import os

import openai
from langchain.chat_models import ChatOpenAI

from src.config import OPENAI_API_KEY
from src.config.functions import FUNCTIONS
from src.services.function_service import FunctionTypeFactory
from src.utils import match_strings

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def openai_chat_model(
    model_name: str,
    streaming: bool = False,
    temperature: float or int = 0.9,
    messages: list or None = None,
    functions: list or None = None,
    function_call: str or None = None,
):
    """Query OpenAI API for a chat model."""
    if functions:
        return openai.ChatCompletion.create(
			messages=messages,
			model=model_name,
			temperature=temperature,
			functions=functions,
			function_call=function_call,
		)
    return openai.ChatCompletion.create(
		messages=messages,
		model=model_name,
		temperature=temperature,
		stream=streaming,
	)

def chat_model(
    model_name: str,
    streaming: bool = False,
    temperature: float or int = 0.9,
    callbacks: list or None = None,
):
    """Query Langchain for a chat model."""
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks,
    )

def openai_chat_functions_model(
	model_name: str,
    streaming: bool = False,
    temperature: float or int = 0.9,
	messages: list or None = None,
    keys: list or None = None,
):
    """Query OpenAI API for a chat model."""
    call_fn = openai_chat_model(
		model_name=model_name,
        messages=messages,
        functions=match_strings(keys, FUNCTIONS),
        function_call="auto",
	)
    response_message = call_fn["choices"][0]["message"]
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_response = FunctionTypeFactory().get_result(
			function_name,
			response_message,
		)
        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append({
			"role": "function",
			"name": function_name,
			"content": function_response,
		})
    return openai_chat_model(
		messages=messages,
		model_name=model_name,
		temperature=temperature,
		streaming=streaming,
	)
        