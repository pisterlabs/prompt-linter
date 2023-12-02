from datetime import datetime
from typing import Optional

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import (
    PromptTemplate
)

class PersonaAgent(BaseModel):
    """기억과 성격, 개성을 가지고 있는 캐릭터"""

    name: str
    current_action: str = 'nothing'
    status: str
    memory = None
    llm: BaseLanguageModel
    verbose: bool
    summary: str
    summary_refresh_seconds: int = 3600
    last_refreshed: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose
        )
    


    