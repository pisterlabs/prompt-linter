import torch
from transformers import Pipeline, pipeline, GenerationConfig, AutoModel, AutoTokenizer
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import List, Optional, Mapping, Any
import sys
import logging

class CPMBeeLLM(LLM):
  max_token: int = 10000
  temperature: float = 0.1
  top_p = .9
  history = []
  tokenizer: object = None
  model: object = None
  model_path:str = None
  device: str = None
  logger: logging.Logger = logging.getLogger()
  
  def __init__(self, model_name_or_path:str, device:str = None):
    super().__init__()
    self.model_path = model_name_or_path
    self.device = "cuda" if torch.cuda.is_available() else "cpu" if device is None or device == "cuda" else device
    
    self.load_model()
    
  @property
  def _llm_type(self) -> str:
    return "ChatGLM"
  
  def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
    response = self.model.generate({"input": prompt, "<ans>":""}, self.tokenizer)
    
    # if stop is not None:
    #   response = enforce_stop_tokens(response, stop)
    self.logger.debug(response)
    
    return response[0]["<ans>"]

  def load_model(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    if self.device == "cuda":
      self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).quantize(4).half().cuda()
    else:
      self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).float()
      
    self.model = self.model.eval()

if __name__ == "__main__":
  torch.cuda.is_available = lambda: False
   
  chatglm = CPMBeeLLM("../models/cpm-bee-10b")

  answer = chatglm.generate(["你好"])
  print(answer)
  