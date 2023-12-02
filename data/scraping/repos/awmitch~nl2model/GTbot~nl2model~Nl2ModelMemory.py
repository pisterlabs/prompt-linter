from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.schema import (
    BaseMessage,
    get_buffer_string,
)
from typing import List
from .Nl2ModelHandler import SummaryHandler


class Nl2ModelMemory(ConversationSummaryMemory):
    summary_callbacks: List[SummaryHandler] = None
    def __init__(self, summary_callbacks=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.summary_callbacks = summary_callbacks
        return
    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        if not self.summary_callbacks:
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
        else:
            chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                callbacks=self.summary_callbacks)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

