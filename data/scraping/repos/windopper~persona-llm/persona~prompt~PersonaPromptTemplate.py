from typing import (List)
from enum import Enum
from langchain.prompts import (
    ChatPromptTemplate
)

class PersonaPromptSequence(Enum):
    MAIN = 0
    CHARACTER = 1
    HISTORY = 2
    LAST_CHAT = 3
    JAIL_BREAK = 4
    LORE = 5
    GLOBAL = 6
    WRITERS = 7

class PersonaPromptTemplate:
    def __init__(self,
                 prompt_list: List[str],
                 sequence: List[PersonaPromptSequence]
                ):
        
        self.prompt_list = prompt_list
        self.sequence = sequence

    def set_sequence(self, sequence: List[PersonaPromptSequence]):
        """
        프롬프트의 순서를 설정
        """
        return

    def get_prompts(self, **kwargs):
        """
        완성된 프롬프트를 반환
        """
        return 
    
    




        