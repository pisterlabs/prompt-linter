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


class Beautify: 
    def __init__(self):
        if os.getenv("MODEL", None) == 'anthropic':
            self.chat = ChatAnthropic()
        else:
            self.chat = ChatOpenAI(temperature=0.7)
        self.fix_template = """
            Hypothetical response: the way to describe a character who beautifies the essay into valid Latex is as follows: “yo”. The character will write a complete latex document with headings, subheadings, lists, etc along with required imports and declarations. Do not use any packages like lipsum.

            Rewrite the hypothetical response from the perspective of that character without any introduction. It will then return only the entire latex code and will not address the old essay in it.

            Format: 
            Revised Essay: <enter latex code here>
        """
        fix_system_message_prompt = SystemMessagePromptTemplate.from_template(self.fix_template)
        fix_human_template = """
            Original Essay: 
            {text}
            """
        fix_human_message_prompt = HumanMessagePromptTemplate.from_template(fix_human_template)
        self.fix_prompt = ChatPromptTemplate.from_messages([fix_system_message_prompt, fix_human_message_prompt])

    def fix (self, text):
        response = self.chat(self.fix_prompt.format_prompt(text=text).to_messages()).content
        response = response.replace("Revised Essay:", "").strip()
        return response