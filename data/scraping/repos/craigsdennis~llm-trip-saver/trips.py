import json
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, messages_from_dict

class CompanyTrip:
    def __init__(self, company, messages):
        print(f"Initializing {company} with {len(messages)} messages")
        self.company = company
        self.messages = messages
        self.chat = ChatOpenAI(model_name=os.environ["OPENAI_MODEL"], temperature=0.7)

    @classmethod
    def from_name(cls, name):
        full_filepath = os.path.join("trips", "companies", name + ".json")
        with open(full_filepath) as file:
            json_str = file.read()
            messages_dict = json.loads(json_str)
            messages = messages_from_dict(messages_dict)
        return cls(name, messages)

    def ask(self, text, store_message=True):
        messages_copy = list(self.messages)
        messages_copy.append(HumanMessage(content=text))
        ai = self.chat(messages_copy)
        if store_message:
            messages_copy.append(ai)
            self.messages = messages_copy
        return ai.content
