from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
import os

class MyQueueHandler(BaseCallbackHandler):
  def __init__(self, q: Queue):
    self.q = q

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.q.put(token)

  def on_llm_end(self, *args, **kwargs) -> None:
    pass

  def on_llm_error(self, error, **kwargs) -> None:
    print(error)

  def empty():
    self.q.empty()


