from typing import Any, Dict
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

class EndpointHandler:
    def __init__(self, path=""):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            return_dict=True,
            device_map="auto",
            # load_in_8bit=True,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        self.pipeline = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, repetition_penalty=1.4, temperature=0.2, max_new_tokens = 512
        )
        
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = data.pop("inputs", data)
        llm = HuggingFacePipeline(pipeline=self.pipeline)
        return llm(prompt)