# @title Vertex AI Utility functions for using with Langchain

import time
from typing import Optional, Tuple, List
from pydantic import BaseModel

from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult
from langchain.llms.vertexai import _VertexAICommon
from langchain.embeddings import VertexAIEmbeddings

from vertexai.preview.language_models import ChatSession


# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
  period = 60 / max_per_minute
  print('Waiting')
  while True:
    before = time.time()
    yield
    after = time.time()
    elapsed = after - before
    sleep_time = max(0, period - elapsed)
    if sleep_time > 0:
      print('.', end='')
      time.sleep(sleep_time)

class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
  requests_per_minute: int
  num_instances_per_batch: int

  # Overriding embed_documents method
  def embed_documents(self, texts: List[str]):
    limiter = rate_limit(self.requests_per_minute)
    results = []
    docs = list(texts)

    while docs:
      # Working in batches because the API accepts maximum 5
      # documents per request to get embeddings
      head, docs = docs[:self.num_instances_per_batch], docs[self.num_instances_per_batch:]
      chunk = self.client.get_embeddings(head)
      results.extend(chunk)
      next(limiter)

    return [r.values for r in results]