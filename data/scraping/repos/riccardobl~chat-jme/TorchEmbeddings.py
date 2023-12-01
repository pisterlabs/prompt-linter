from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from sentence_transformers import SentenceTransformer, models
import torch
import numpy as np
import threading
from torch import nn
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample

class TorchEmbeddings( BaseModel,Embeddings):
    
    
    @staticmethod
    def init(device):
        isGpuDevice=device=="cuda" or device=="gpu"
        if isGpuDevice and  not torch.cuda.is_available():
            print("WARNING: GPU device requested but not available")

        model_name='sentence-transformers/all-mpnet-base-v2'
        #model_name='sentence-transformers/paraphrase-MiniLM-L6-v2'
        print("Loading "+model_name+" model...")
        TorchEmbeddings.torch_device='cuda' if isGpuDevice and torch.cuda.is_available() else 'cpu'

        TorchEmbeddings.model = SentenceTransformer(model_name,device=TorchEmbeddings.torch_device)          
        TorchEmbeddings.model.max_seq_length=512
   
        
        print("Done")

  
    def _embedding_func2(self, texts):
        torch_device=TorchEmbeddings.torch_device
        model=TorchEmbeddings.model
        texts = [text.replace("\n", " ").lower()  for text in texts]
      
        embeddings = model.encode(
            texts,
            device=torch_device,
            show_progress_bar=True,
            convert_to_numpy=True
        )        
        return embeddings

           
    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        return self._embedding_func2([text])[0]


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        torch_device=TorchEmbeddings.torch_device
        model=TorchEmbeddings.model
        responses = self._embedding_func2(texts)
        return responses

    def embed_query(self, text: str) -> List[float]:
        torch_device=TorchEmbeddings.torch_device
        model=TorchEmbeddings.model
        embedding = self._embedding_func(text, engine=torch_device)
        return embedding
