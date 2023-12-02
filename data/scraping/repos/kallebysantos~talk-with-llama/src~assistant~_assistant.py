import hashlib
from typing import Any, Dict, Optional

from langchain import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from .streaming_web import StreamingWebCallbackHandler

class Assistant():
    model: BaseLanguageModel
    chains: Dict[str, LLMChain] = {}
    handler = StreamingWebCallbackHandler()

    def __init__(self, model_path: str):
        pass
        
    def new_chain(self, **kwargs: Any) -> LLMChain: 
        pass
    
    def add_chain(self, key: str, **kwargs: Any):
        hashed_key = hashlib.sha256(str.encode(key)).hexdigest()

        if(hashed_key not in self.chains):
            self.chains[hashed_key] = self.new_chain(**kwargs)

        return self.chains[hashed_key], hashed_key

    def get_chain(self, hashed_key: str) -> Optional[LLMChain]:
        return self.chains[hashed_key] if hashed_key in self.chains else None