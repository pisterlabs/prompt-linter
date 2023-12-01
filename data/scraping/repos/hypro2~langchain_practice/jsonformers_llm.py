import json
from functools import partial
from typing import List, Mapping, Optional, Any, Dict

import jsonformer
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class JSONformersLLM(LLM):
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    backend: Optional[str] = 'llama'
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 200
    repetition_penalty: Optional[float] = 1.15
    ## 추가 ##
    model: Any = None
    tokenizer: Any = None

    #########

    def __init__(self, model_folder_path, callbacks=None, **kwargs):
        super(JSONformersLLM, self).__init__()
        self.model_folder_path: str = model_folder_path
        self.callbacks = callbacks

        ## 추가 ##
        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True,
                                                          do_sample=True,
                                                          device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder_path, use_fast=False)
        #########

    @property
    def _get_model_default_parameters(self):
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    @property
    def _llm_type(self) -> str:
        return 'llama'

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              json_schema: Dict = None,
              **kwargs) -> str:

        params = {
            **self._get_model_default_parameters,
            **kwargs
        }

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        ## 추가 ##
        model = jsonformer.Jsonformer(model=self.model,
                                      tokenizer=self.tokenizer,
                                      json_schema=json_schema,
                                      prompt=prompt,
                                      max_array_length=params['max_tokens'],
                                      max_number_tokens=params['max_tokens'],
                                      max_string_token_length=params['max_tokens'],
                                      temperature=params['temperature']
                                      )
        text = model()
        if stop:
            text = enforce_stop_tokens(text, stop)
        #######

        return json.dumps(text)
