from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    Union,
)
import numpy as np
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
import  requests
from dotenv import load_dotenv
load_dotenv()
import os
class VicunaEmbeddings(BaseModel, Embeddings):
    uid: str = None
    server_url:str = None
    client: Any  #: :meta private:
    model: str = "vicuna-13b"
    embedding_ctx_length: int = 8191
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""


    def _embed(self, texts: Union[List[str],str]):
        headers = {"Content-Type": "application/json"}
        if isinstance(texts, str):
            texts = [texts]
        
        response = requests.post(
            self.server_url+"/worker_get_embeddings",
            json={
                "input": texts,
                "model": "vicuna-13b",
            },
            headers=headers,
        )
        try:
            if response.status_code == 200:
                json_data = response.json()
                return json_data["embedding"]   
        except:
            if response.status_code != 200:
                print(response.content)
            else:
                print("Data formatting incorrect")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Hosted OpenAIs Document endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)
        

    def embed_query(self, text: str) -> List[float]:
        """Call out to  Hosted OpenAIs, query embedding endpoint
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        return self._embed(text)