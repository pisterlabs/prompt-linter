from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory

from app.web.api import (
  get_messages_by_conversation_id, 
  add_message_to_conversation
)

class SqlMessageHistory(BaseChatMessageHistory):
  conversation_id: str
  
  def __init__(self, conversation_id) -> None:
    self.conversation_id = conversation_id
  
  @property
  def messages(self):
    return get_messages_by_conversation_id(
      self.conversation_id
    )
  
  def add_message(self, message):
    return add_message_to_conversation(
      conversation_id=self.conversation_id,
      role=message.type,
      content=message.content
    )
    
  def clear(self):
    pass
    

def build_memory(chat_args):
  return ConversationBufferMemory(
    chat_memory=SqlMessageHistory(
      chat_args.conversation_id,
    ),
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
  )
