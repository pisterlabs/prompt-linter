from langchain.embeddings.base import Embeddings
import time
import requests
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain

class YaGPTEmbeddings(Embeddings):

    def __init__(self,folder_id,api_key,sleep_interval=1, retries=3):
        self.folder_id = folder_id
        self.api_key = api_key
        self.sleep_interval = sleep_interval
        self.headers = { 
                        "Authorization" : f"Api-key {api_key}",
                        "x-folder-id" : folder_id }
        self.retries = retries
        
    def embed_document(self, text):
        j = {
          "model" : "general:embedding",
          "embedding_type" : "EMBEDDING_TYPE_DOCUMENT",
          "text": text
        }
        r = self.retries
        while True:
            res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                                json=j,headers=self.headers)
            js = res.json()
            if 'embedding' in js:
                return js['embedding']
            r-=1
            if r==0:
                raise Exception(f"Cannot process embeddings, result received: {js}")
            time.sleep(self.sleep_interval)

    def embed_documents(self, texts, chunk_size = 0):
        res = []
        for x in texts:
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res
        
    def embed_query(self, text):
        j = {
          "model" : "general:embedding",
          "embedding_type" : "EMBEDDING_TYPE_QUERY",
          "text": text
        }
        r = self.retries
        while True:
            res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                                json=j,headers=self.headers)
            js = res.json()
            if 'embedding' in js:
                return js['embedding']
            r-=1
            if r==0:
                raise Exception(f"Cannot process embeddings, result received: {js}")    
            time.sleep(self.sleep_interval)

class YandexLLM(langchain.llms.base.LLM):
    api_key: str = None
    iam_token: str = None
    folder_id: str
    max_tokens : int = 1500
    temperature : float = 1
    instruction_text : str = None
    sleep_interval : float = 1
    retries : int = 3

    @property
    def _llm_type(self) -> str:
        return "yagpt"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        headers = { "x-folder-id" : self.folder_id }
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        if self.api_key:
            headers["Authorization"] = f"Api-key {self.api_key}"
        req = {
          "model": "general",
          "instruction_text": self.instruction_text,
          "request_text": prompt,
          "generation_options": {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
          }
        }
        r = self.retries
        while True:
            res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
                    headers=headers, json=req)
            js = res.json()
            if 'result' in js:
                return js['result']['alternatives'][0]['text']
            r-=1
            if r==0:
                raise Exception(f"Cannot process YaGPT request, result received: {js}")
            time.sleep(self.sleep_interval)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_tokens": self.max_tokens, "temperature" : self.temperature }
