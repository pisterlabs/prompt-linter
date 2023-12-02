from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
from typing import Optional, List
from langchain.callbacks.manager import CallbackManager
import os
from langchain import PromptTemplate, LLMChain
from .model_configs import get_model
from tqdm import trange, tqdm
import time

@dataclass_json
@dataclass
class MultiModelPrompterConfig:
    dotenv_path: str = ".env"
    gpt4_32k: bool = False
    gpt4_8k: bool = False
    gpt35_turbo: bool = False
    text_davinci: bool = False
    # bits: 1111 = (7B, 13B, 30B, 65B)
    llama: str = False
    # bits: 1111 = (7B, 13B, 30B, 65B)
    alpaca: str = False
    
    # will add this at a later date
    gpt4all_j: bool = False


class MultiModelPrompter:
    def __init__(self, config: MultiModelPrompterConfig):        
        self.config = config
        
        # using these model names because I can't load everything at once
        # there is a factory get_model which will give us the models at runtime
        self.model_names = []
        
        if self.config.gpt4_32k:
            self.model_names.append("gpt4-32k")
            
        if self.config.gpt4_8k:
            self.model_names.append("gpt4-8k")
            
        if self.config.gpt35_turbo:
            self.model_names.append("gpt35-turbo")
            
        if self.config.text_davinci:
            self.model_names.append("text-davinci")
        
        if self.config.llama:
            if len(self.config.llama) != 4:
                raise AttributeError("There are only 4 llama models! String should be bitmap of 0 | 1")
            
            bool_map = ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]
            for i in range(len(bool_map)):
                if self.config.llama[i] == "1":
                    self.model_names.append(bool_map[i])
                    
        if self.config.alpaca:
            if len(self.config.alpaca) != 4:
                raise AttributeError("There are only 4 llama models! String should be bitmap of 0 | 1")
            
            bool_map = ["alpaca-7b", "alpaca-13b", "alpaca-30b", "alpaca-65b"]
            for i in range(len(bool_map)):
                if self.config.llama[i] == "1":
                    self.model_names.append(bool_map[i])
                    
        if self.config.gpt4all_j:
            raise NotImplementedError
        
        if len(self.model_names) == 0:
            raise AttributeError
        
    def run_prompt(self,
                   output_path: str,
                   prompts: List[tuple[int, str]],
                   prompt_templates: List[PromptTemplate],
                   callback_manager: CallbackManager,
                   max_tokens: int=256,
                   verbose: bool = False,
                   dotenv_path: str = "apidata.env",
                   num_repeats: int = 1):
        """
        NOTE: Format for prompts should be (PromptTemplateIndex, PromptString)
        """
        for name in tqdm(self.model_names):
            model = get_model(name, max_tokens, callback_manager, verbose, dotenv_path)
            
            prompt_len = len(prompts)
            for idx, prompt in tqdm(prompts, leave=False):
                assert idx < len(prompt_templates)
                template = prompt_templates[idx]
                
                llm_chain = LLMChain(prompt=template, llm=model)
                
                for _ in trange(0, num_repeats, leave=False):
                    with open(output_path, "a") as f:
                        chain_resp = f"Model: {name}\nTemplate: {idx}\nQuestion: {prompt}\n\n"
                        
                        for _ in range(5): # try 5 times for the rate limit
                            try:
                                chain_resp += llm_chain.run(prompt)
                                chain_resp += "\n----------------------------------------------------------------\n"
                                f.write(chain_resp)
                                break
                            except Exception as e:
                                chain_resp += str(e)
                                time.sleep(15)
                        else:
                            raise TimeoutError("Rate limited more than 5 times! Exiting.")
                    
            