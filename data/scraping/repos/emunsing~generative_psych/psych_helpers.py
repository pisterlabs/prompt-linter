from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import tiktoken
import textwrap
from langchain.callbacks import get_openai_callback

import logging

LOGGER = logging.getLogger(__name__)

class OpenAIQueryAPI:
    def __init__(self, modeltype='gpt-3.5-turbo-0613', temperature='0.5') -> None:
        self.llm = ChatOpenAI(model_name=modeltype, temperature=temperature)
        
    def __call__(self, prompt):
        system_prompt = textwrap.dedent("""\
            You are an excellent writer developing a screenplay based on the following prompt. The dialogue should be realistic, 
            descriptive, nuanced, and should feel true to real life. Writing a realistic depiction is important, even if it means that 
            characters get mad at each other, show strong emotions, and get upset.
            """).replace("\n", " ")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
    ]
        LOGGER.debug(f"Query: {prompt}")
        with get_openai_callback() as cb:
           res = self.llm(messages).content

        LOGGER.info(f"Token count: {cb.total_tokens}")
        LOGGER.debug(f"Response: {res}")

        return res
