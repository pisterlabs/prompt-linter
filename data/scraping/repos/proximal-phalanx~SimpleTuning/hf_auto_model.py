from typing import List, Tuple, Any, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from accelerate import Accelerator

from langchain.llms.base import LLM
from simpletuning import *
from transformers import LlamaTokenizer
import pickle

import os

import json

from transformers.generation.configuration_utils import GenerationConfig


class HuggingfaceAutoModel(LLMInterface):
    """
    This is for transformers models without chat function.
    Use generate instead.
    """

    tokenizer: Any
    model: Any
    model_name_or_path: str
    bnb_config: Optional[BitsAndBytesConfig]
    acc: Any

    def __init__(self, model_name_or_path, bnb_config=None) -> None:
        self.model_name_or_path = model_name_or_path
        self.bnb_config = bnb_config

    def load(self) -> None:
        if self.bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                max_length=256,
                quantization_config=self.bnb_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True, max_length=256
            )
        self.model = self.model.cuda()
        if self.model.config.model_type.lower() == "llama":
            # Due to the name of transformers' LlamaTokenizer, we have to do this
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        self.acc = Accelerator()
        self.model, self.tokenizer = self.acc.prepare(self.model, self.tokenizer)
        # self.model.generation_config = GenerationConfig.from_pretrained(
        #     self.model_name_or_path, trust_remote_code=True
        # )

    def query(
        self, prompt: str, **kwargs
    ) -> str:
        """
        history is ignored
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response

    # def dump(self, path: str) -> None:
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     config = {
    #         "model_name_or_path": convert_to_hf(self.model_name_or_path),
    #         "use_half": self.use_half,
    #     }
    #     pickle.dump(
    #         config, open(os.path.join(path, LLM_INTERFACE_CONFIG_FILE_NAME), "wb")
    #     )
    #     pickle.dump(
    #         self.__class__,
    #         open(os.path.join(path, LLM_INTERFACE_CLASS_FILE_NAME), "wb"),
    #     )

    # @classmethod
    # def load_dump(cls, path: str) -> "HuggingfaceAutoModel":
    #     config = pickle.load(
    #         open(os.path.join(path, LLM_INTERFACE_CONFIG_FILE_NAME), "rb")
    #     )
    #     return cls(**config)
