from langchain import LLMChain
from user_template import USER_GENERATOR_TEMPLATE
from scai.chat_models.crfm import crfmChatLLM
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

class UserGenerator():
    """
    User Generator Class
    """
    def __init__(self, template: ChatPromptTemplate):
        self.template = template
        self.llm = crfmChatLLM(model_name="openai/gpt-4-0314", max_tokens=150, temperature=0.9)
        self.chain = LLMChain(llm=self.llm, prompt=self.template, memory=None)

class TaskConGenerator():
    """
    Task Connectives Generator Class
    """
    def __init__(self, template: ChatPromptTemplate):
        self.template = template
        self.llm = crfmChatLLM(model_name="openai/gpt-4-0314", max_tokens=150, temperature=0.9)
        self.chain = LLMChain(llm=self.llm, prompt=self.template, memory=None)

class UserTaskConGenerator():
    """
    Generates A User as well as its Connectives
    """
    def __init__(self):
        self.user_data = USER_GENERATOR_TEMPLATE["user_template_1"]
        self.user_prompt, self.task_con_prompt = self.user_data.content_user_gen, self.user_data.content_user_con

    def get_prompt(self, str: str) -> ChatPromptTemplate:
        if str == "user":
            prompt_template = self.user_prompt
        else:
            prompt_template = self.task_con_prompt
        system_prompt_template = SystemMessagePromptTemplate.from_template("Always respond to the best of your ability.\n")
        return ChatPromptTemplate.from_messages([system_prompt_template, prompt_template])

    def create_user(self, attributes: str) -> str:
        chat_prompt_template = self.get_prompt("user")
        user_generator = UserGenerator(template=chat_prompt_template)
        user = user_generator.chain.run(attributes=attributes, stop=["System:"])
        return user

    def create_task_con(self, task_attributes: str, user: str, task: str) -> str:
        chat_prompt_template = self.get_prompt("task_connectives")
        task_con_generator = TaskConGenerator(template=chat_prompt_template)
        if task:
            task_con = task_con_generator.chain.run(user=user, task_attributes=task_attributes, question=task, stop=["System:"])
            return task_con
        else:
            raise Exception("No task specified!")