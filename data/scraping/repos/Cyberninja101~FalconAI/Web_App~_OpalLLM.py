#basic LangChain LLM wrapper for OPAL
from langchain.llms.base import BaseLLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
import json
import requests
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Union, Dict

class OpalLLM(LLM):
    model: str
    bad_word_list: Union[None, List[List[str]]] = None
    stop_word_list: Union[None, List[List[str]]] = [["</s>"]]
    max_tokens: Union[int, None] = None
    top_k: Union[None, int] = None
    top_p: Union[float, None] = None
    temperature: Union[float, None] = 0.0
    repetition_penalty: Union[float, None] = None
    random_seed: Union[None, int] = None
    base_url: str = "https://opal.livelab.jhuapl.edu:8080/completions"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:

        response = self.opal_ask(prompt)
        # return response
        txt: str = response["choices"][0]["text"]
        output_only: str = self._postprocess_response(prompt, txt)
        return output_only
    

    def _postprocess_response(self, input_prompt: str, resp: str) -> str:
        output_only: str = resp.replace(input_prompt, "").replace("<s>", "")
        if "\n### Human" in output_only:
            return output_only.split("\n### Human")[0]
        elif "Input: <noinput>" in output_only:
            return output_only.split("\nInput:")[0]
        return output_only
    
    @property
    def _llm_type(self) -> str:
        return "opal-llm-model"

    
    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for API"""
        defaults = {
            "model": self.model,
            "bad_word_list": self.bad_word_list,
            "stop_word_list": self.stop_word_list,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "random_seed": self.random_seed,
        }
        # remove none values
        return {k: v for k, v in defaults.items() if v != None}
    def opal_ask(self, prompt):
        response = requests.post(self.base_url, 
                                #headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                             json={
                                        "model": self.model,
                                        "prompt": prompt,
                                        "max_tokens": self.max_tokens,
                                        "temperature": self.temperature,
                                        "top_p": 1,
                                        "n": 1,
                                        "bad_word_list": [
                                            [
                                            ""
                                            ]
                                        ],
                                        "stop_word_list": [
                                            [
                                            ""
                                            ]
                                        ],
                                        "repetition_penalty": 1.2,
                                        "random_seed": -1,
                                        "top_k": 50,
                                        "client_type": "http"
                                        }
                                        )
        if response.ok:
            return response.json()
        else:
            print(response.text)
            return None