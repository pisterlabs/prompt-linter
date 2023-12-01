from typing import List, Tuple, Any, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers.generation import GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import LlamaTokenizer
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from simpletuning import *
import os

import pickle
import json

from accelerate import Accelerator


class HuggingfaceAutoModelWithAdapter(LLMInterface):
    """
    This is for model with model.
    """

    tokenizer: Any
    model: Any
    acc: Any
    model_name_or_path: str
    bnb_config: Optional[BitsAndBytesConfig]
    adpater_path: str

    def __init__(
        self,
        model_name_or_path: str,
        adpater_path: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adpater_path = adpater_path
        self.bnb_config = bnb_config

    def load(self) -> None:
        """
        max_length will only be used when there is not a generation_config.json in the model directory.
        """
        device_map = "auto"
        # if os.path.exists(
        #     os.path.join(self.model_name_or_path, "generation_config.json")
        # ):
        #     base = AutoModelForCausalLM.from_pretrained(
        #         self.model_name_or_path,
        #         trust_remote_code=True,
        #     )
        #     self.model = PeftModel.from_pretrained(
        #         base,
        #         self.adpater_path,
        #     )
        #     self.model.generation_config = GenerationConfig.from_pretrained(
        #         self.model_name_or_path, trust_remote_code=True
        #     )
        #     if self.use_half:
        #         self.model = self.model.half()
        # else:
        if self.bnb_config:
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                max_length=256,
                quantization_config=self.bnb_config,
                device_map=device_map,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                max_length=256,
                device_map=device_map,
            )
        self.model = PeftModel.from_pretrained(
            base,
            self.adpater_path,
            device_map=device_map
        )
        if self.model.config.model_type.lower() == "llama":
            # Due to the name of transformers' LlamaTokenizer, we have to do this
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        self.acc = Accelerator()
        self.model, self.tokenizer = self.acc.prepare(self.model, self.tokenizer)

    def query(
        self, prompt: str, **kwargs
    ) -> str:
        """
        History will be integrated into the prompt.
        """
        # prompt = "\n".join([f"User:{p}\nYou:{o}\n" for p, o in history]) + "\n" + prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

        output_ids = self.model.generate(inputs=input_ids)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
