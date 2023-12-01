from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class FormattedPromptTemplate:
    def __init__(self, action_template: str, format: str):
        action_template +="""
        You are not verbose, 
        You always get straight to the point, your format always remains consistent,
        You do not repeat yourself.
        You only output in the format, nothing else, if json you output only json, if xml only xml, if react only react
        and so on for all formats.
        """
        self.action_template = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(action_template), 
                                                        HumanMessagePromptTemplate.from_template("{input}")])
        self.format = format
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
