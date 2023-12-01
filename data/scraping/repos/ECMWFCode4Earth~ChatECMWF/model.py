"""
This module loads the models and the utilities for running it. It is possible to enable the log to stdout of all LLM interactions
by controlling the ```langchain.debug``` parameter. A subclass of ```langchain.memory.ConversationBufferMemory``` has been implemented for
coping with different output keys from the tools. In particular the RetrievalQA tools and the StructuredTools have different output, which need to be parsed with some logic. This module allows the replacement of the LLM simply by replacing the ```model`` object. See the LLama2 branch for a practical example.
"""
import langchain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from .config import Logger, configs

langchain.deubg = configs.DEBUG


class MyBuffer(ConversationBufferMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_context(self, inputs, outputs):
        if "source_documents" in outputs["output"]:
            self.output_key = "result"
            outputs = outputs["output"]
        else:
            self.output_key = "output"
        super().save_context(inputs, outputs)


chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = MyBuffer(
    memory_key="chat_history",
    input_key="input",
    output_key="output",
    return_messages=True,
)

llm = OpenAI(
    model_name=configs.GPT_VERSION,
    temperature=configs.GPT_TEMPERATURE,
    max_tokens=configs.MAX_TOKENS,
)
