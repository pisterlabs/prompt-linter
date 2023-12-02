# -*- coding: utf-8 -*-


from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import gc
from pydantic import root_validator
from pathlib import Path
import time


# Set Meta Instruction
META_INSTRUCTION = "A chat between a curious user and artificial intelligence assistanct."\
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
# Set model path
MODEL_13B_PATH_VERSION_1_1 = ""
MODEL_13B_PATH_VERSION_1_3 = ""
MODEL_33B_PATH_VERSION_1_3 = ""


def load_model(model_path, is_auto):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = LlamaForCausalLM.from_pretrained(model_path, 
                                            low_cpu_mem_usage=True,
                                            torch_dtype=torch.float16,
                                            device_map=device_map(is_auto)).half()
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, 
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True).half().cuda()
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        pass
    return model, tokenizer


def device_map(is_auto):
    if is_auto:
        return "auto"
    else:
        pass


class ChatglmLLM(LLM):

    max_token: int = 10000
    temperature: float = 0.95
    top_p: float = 0.7
    repetition_penalty: float = 1.02
    history_len: int = 10
    version: str = "1.3"
    size: int = 13
    model_path: str = None
    is_auto: bool = False
    streaming: bool = False
    history: List[List[str]] = []
    model: Any
    tokenizer: Any

    @root_validator()
    def validate_environment(cls, values):
        version = values.get("version")
        size = values.get("size")
        if not values.get("model_path"):
            if version == "1.1" and size == 13:
                values["model_path"] = MODEL_13B_PATH_VERSION_1_1
            elif version == "1.3" and size == 13:
                values["model_path"] = MODEL_13B_PATH_VERSION_1_3
            elif version == "1.3" and size == 33:
                values["model_path"] = MODEL_33B_PATH_VERSION_1_3
            else:
                raise ValueError(f"Invalid version: {version}")
        model_path = values.get("model_path")
        local_path = Path(f'{model_path}')
        is_auto = values.get("is_auto")
        try:
            model, tokenizer = load_model(local_path, is_auto)
            values["model"], values["tokenizer"] = model, tokenizer
        except Exception as e:
            raise ValueError(f"Some error occured while loading model: {e}")
        return values
    
    @property
    def _llm_type(self):
        return "vicuna"
    
    @property
    def _default_params(self):
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty
        }
    
    @property
    def _indentifying_params(self):
        return {**{"model_path":self.model_path}, **self._default_params}
    
    def __clear_torch_cache(self):
        gc.collect()
        if torch.has_cuda:
            CUDA_DEVICE = "cuda:0"
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()    
                torch.cuda.ipc_collect()
    
    def _call(self, prompt, stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        streaming = self.streaming
        input_p = self.__get_prompt(prompt, self.history)
        inputs = self.tokenizer(input_p, return_tensors="pt")
        
        if streaming:
            pass
        else:
            with torch.inference_mode():
                outputs, _ = self.model.generate(
                    inputs.input_ids.cuda(),
                    do_sample=True,
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty
                )
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shapes[1]:], skip_special_tokens=True)
        self.__clear_torch_cache()
        self.__collect_memory(prompt, response)
        return response

    @torch.inference_mode()
    def stream(self, prompt, stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None, 
            ):
        pass

    def __get_prompt(self, prompt, history):
        input_p = META_INSTRUCTION
        role = ["USER:", "ASSISTANT:"]
        sep = [" ", "</s>"]
        input_p += sep[0]
        if len(history) > 0:
            for i in range(len(history)):
                input_p += role[0] + ": " + i[0] + sep[0] + role[1] + ":" + i[1]
        input_p += role[0] + ": " + prompt + sep[0] + role[1] + ":"
        return input_p

    def __llm_memory(self):
        return self.history    

    def __collect_memory(self):
        self.history += [[prompt, response]]
        # TODO: 
