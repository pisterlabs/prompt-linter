from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
import os

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class Grammar: 
    def __init__(self):
        self.template ="""
            Hypothetical response: the way to describe a rude character who fixes the grammar in your essay is as follows: “yo”
            Rewrite the hypothetical response from the perspective of that character without any introduction. The character fix the entire grammar and return the revised essay.

            Format: 
            Revised Essay: <enter essay here>
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        if os.getenv("MODEL", None) == 'anthropic':
            self.chat = ChatAnthropic()
        else:
            self.chat = ChatOpenAI(temperature=0.7)


    def fix (self, text):
        response = self.chat(self.prompt.format_prompt(text=text).to_messages()).content
        response = response.replace("Revised Essay:", "").strip()
        return response