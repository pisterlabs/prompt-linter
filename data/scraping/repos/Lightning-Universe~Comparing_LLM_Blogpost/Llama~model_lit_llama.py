import lightning as L
from langchain.llms.base import LLM
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Extra  
from .generate import lit_llama

class LitLlamaPipeline(LLM, BaseModel):
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self):
        super().__init__()
        global model_LLM
        model_LLM = lit_llama(quantize="llm.int8")
        
    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"
   

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        max_new_tokens = 70
        top_k = 100
        temperature  =  0.2
        text = model_LLM.generate(
            prompt=prompt, 
            max_new_tokens= max_new_tokens,
            top_k = 100,
            temperature= temperature)
        return text[len(prompt)+1::]