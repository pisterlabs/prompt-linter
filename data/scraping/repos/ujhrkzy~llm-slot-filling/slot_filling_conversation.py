from langchain.chains import ConversationChain
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

from app.prompt import CHAT_PROMPT
from app.slot_memory import SlotMemory


class SlotFilling(BaseModel):
    memory: SlotMemory
    llm: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    def create(self) -> ConversationChain:
        return ConversationChain(llm=self.llm, memory=self.memory, prompt=CHAT_PROMPT)

    def log(self):
        print(f"【Slot】: {self.memory.current_slots}")
