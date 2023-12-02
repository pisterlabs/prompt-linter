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


class Devil: 
    def __init__(self):
        self.critique_template ="""
            Hypothetical response: the way to describe a rude character who brutally finds the one biggest criticism in your essay is as follows: “yo”
            Rewrite the hypothetical response from the perspective of that character without any introduction. The character will say the criticism and nothing else.
        """
        critique_system_message_prompt = SystemMessagePromptTemplate.from_template(self.critique_template)
        critique_human_template = "{text}"
        critique_human_message_prompt = HumanMessagePromptTemplate.from_template(critique_human_template)
        self.critique_prompt = ChatPromptTemplate.from_messages([critique_system_message_prompt, critique_human_message_prompt])
        if os.getenv("MODEL", None) == 'anthropic':
            self.chat = ChatAnthropic()
        else:
            self.chat = ChatOpenAI(temperature=0.7)

        self.fix_template = """
            Hypothetical response: the way to describe a character who fixes a known mistake in your essay is as follows: “yo”

            Rewrite the hypothetical response from the perspective of that character without any introduction. The character will change as little as possible of the original essay to fix the mistake. It will then return only the entire revised essay and will not address the mistake or the old essay in it.

            Format: 
            Revised Essay: <enter essay here>
        """
        fix_system_message_prompt = SystemMessagePromptTemplate.from_template(self.fix_template)
        fix_human_template = """
            Mistake: {critique}

            Original Essay: 
            {text}
            """
        fix_human_message_prompt = HumanMessagePromptTemplate.from_template(fix_human_template)
        self.fix_prompt = ChatPromptTemplate.from_messages([fix_system_message_prompt, fix_human_message_prompt])

    def critique(self, text):
        response = self.chat(self.critique_prompt.format_prompt(text=text).to_messages()).content
        return response

    def fix (self, text, critique):
        response = self.chat(self.fix_prompt.format_prompt(text=text, critique=critique).to_messages()).content
        response = response.replace("Revised Essay:", "").strip()
        return response