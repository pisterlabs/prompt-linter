from langchain.llms.base import LLM
from typing import Any, Mapping, Optional, List

import requests


API_URL = "https://api-inference.huggingface.co/models/togethercomputer/GPT-JT-6B-v1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

class CustomLLM(LLM):
    
    n: int
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        response = query({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.2,
                "return_full_text": False,
            },
            # "options": {
            #     "wait_for_model": True,
            # }
        })
        print("RESPONSE: ", response)
        return response[0]["generated_text"]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    