from langchain.embeddings.base import Embeddings
import time
import requests
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain

class YandexGPTEmbeddings(Embeddings):

    def __init__(self, iam_token=None, api_key=None, folder_id=None, sleep_interval=1):
        self.iam_token = iam_token
        self.sleep_interval = sleep_interval
        self.api_key = api_key
        self.folder_id = folder_id
        if self.iam_token:
            self.headers = {'Authorization': 'Bearer ' + self.iam_token}
        if self.api_key:
            self.headers = {'Authorization': 'Api-key ' + self.api_key,
                             "x-folder-id" : self.folder_id }
                
    def embed_document(self, text):
        j = {
          "model" : "general:embedding",
          "embedding_type" : "EMBEDDING_TYPE_DOCUMENT",
          "text": text
        }
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                            json=j, headers=self.headers)
        vec = res.json()['embedding']
        return vec

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
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                            json=j,headers=self.headers)
        vec = res.json()['embedding']
        time.sleep(self.sleep_interval)
        return vec
    


class YandexLLM(langchain.llms.base.LLM):
    api_key: str = None
    iam_token: str = None
    folder_id: str = None
    max_tokens : int = 1500
    temperature : float = 1
    instruction_text : str = None

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
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
          headers=headers, json=req).json()
        return res['result']['alternatives'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_tokens": self.max_tokens, "temperature" : self.temperature }
