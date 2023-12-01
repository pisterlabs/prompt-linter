from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List 

class Word(BaseModel):
    word: str = Field(description='the word itself')
    level: str = Field(description='level of vocabulary')
    classification: str = Field(description='either the word is formal, informal or business')
    meaning: List[str] = Field(description='meaning of the word')
    synonyms: List[str] = []
    antonyms: List[str] = []
    examples: List[List] = Field(description='list of examples of the usage of the word')
