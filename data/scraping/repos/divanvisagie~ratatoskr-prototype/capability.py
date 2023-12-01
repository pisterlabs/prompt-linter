import os
from typing import List
from clients.openai_client import get_text_answer
from capability.duck_duck_go.capability import DuckDuckGoCapability
from capability.capability import Capability
from capability.chat_gpt.capability import ChatGptCapability
from message_handler.message_types import RequestMessage, ResponseMessage

def get_prompt(question, filters: str):
    prompt = f"""I have a list of classes that perform different tasks for a user

    {filters}

    Given the following conversation:

    User: {question}

    In one word with no punctuation, which filter should be used?"""

    return prompt

def get_target_filter(text: str, filters: List[Capability]):
    filter_str = build_filter_list(filters)
    prompt = get_prompt(text, filter_str)
    answer = get_text_answer(prompt).strip()
    return answer

def filter_to_description(filter: Capability):
    return f'{filter.__class__.__name__}: {filter.description}'

def build_filter_list(filters: List[Capability]):
    descriptions =  [filter_to_description(filter) for filter in filters]
    return os.linesep.join(descriptions)

class SmartSwitchFilter(Capability):
    """Uses response from openai to determine which filter to use"""
    def __init__(self):
        super().__init__()
        self.filters: List[Capability] = [
            # NotionFilter(),
            DuckDuckGoCapability(),
            ChatGptCapability([])
        ]
    
    def relevance_to(self, msg: RequestMessage):
        return 0.0
    
    def apply(self, msg: RequestMessage) -> ResponseMessage:
        filter = get_target_filter(msg.text, self.filters)
        for f in self.filters:
            if f.__class__.__name__ == filter:
                return f.apply(msg)

        return  ResponseMessage("Unfortunately I cant find anything to do", self.__class__.__name__)