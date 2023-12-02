from typing import Any, List, Optional, Union
import os, openai
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from agent_sim.prompts_library import INPUT_PROMPT, REFLECT_USER_PROMPT, REFLECT_SYSTEM_PROMPT

class Player:
    def __init__(
        self, respond_model: BaseLanguageModel, reflect_model: BaseLanguageModel, inception_prompt: str, role_name: str, max_context_length: int = 1000) -> None:
        self.respond_model = respond_model
        self.reflect_model = reflect_model
        self.inception_prompt = inception_prompt
        self.role_name = role_name
        self.max_context_length = max_context_length
        self.memory: List[str] = []
        self.memory_length: int = 0

    def respond(self, input_role: str, input_message: str, remember: bool = True) -> Union[str, Any]:
        human_prompt = INPUT_PROMPT.format(role_name=self.role_name, history="\n".join(self.memory), message=input_message, input_role=input_role)

        prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(self.inception_prompt), HumanMessagePromptTemplate.from_template(human_prompt)]).format_messages(memory=self.memory)

        human_prompt = INPUT_PROMPT.format(
            role_name=self.role_name,
            history="\n".join(self.memory),
            message=input_message,
            input_role=input_role,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.inception_prompt),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        ).format_messages(memory=self.memory)

        response = self.respond_model.predict_messages(
            prompt, tags=[self.role_name, "respond"]
        ).content

        if remember:
            self.add_to_memory(input_role, input_message)
            self.add_to_memory(self.role_name, response)
        return response


    def add_to_memory(self, role: str, message: str) -> None:
        message = f"{role}: {message}"
        self.memory.append(message)
        self.memory_length += len(message)
        if self.memory_length >= self.max_context_length:
            self.reflect()

    def reflect(self) -> None:
        num_messages = min(10, len(self.memory) - 2)
        messages_to_process = "\n".join(self.memory[:num_messages])
        processed_messages = self.reflect_model.predict_messages([SystemMessage(content=REFLECT_SYSTEM_PROMPT.format(role_name=self.role_name)), HumanMessage(content=REFLECT_USER_PROMPT.format(history=messages_to_process))], tags=[self.role_name, "reflect"]).content

        self.memory = [processed_messages] + self.memory[num_messages:]

        self.memory_length = sum(len(message) for message in self.memory)