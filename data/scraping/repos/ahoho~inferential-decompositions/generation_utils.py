from typing import Optional, Any

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

import langchain
from langchain.schema import LLMResult, Generation
from langchain.llms.utils import enforce_stop_tokens

from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    pipeline,
)


class OpenAIPB(langchain.OpenAI):
    def get_sub_prompts(self, params: dict[str, Any], prompts: list[str], stop: Optional[list[str]] = None) -> list[list[str]]:
        return tqdm(super().get_sub_prompts(params, prompts, stop))


class OpenAIChatPB(langchain.llms.OpenAIChat):
    def modelname_to_contextsize(self, modelname: str) -> int:
        return 4096
    
    def generate(self, prompts: list[str], stop: Optional[list[str]] = None) -> list[LLMResult]:
        generations = []
        for messages, instance in tqdm(prompts):
            self.prefix_messages = messages
            generations.append(super().generate([instance], stop))
        self.prefix_messages = [] # clear
        return generations


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class HuggingFacePipelineDS(langchain.HuggingFacePipeline):
    """Use datasets to generate prompts, add progress bar"""
    def _generate(self, prompts: list[str], stop: Optional[list[str]] = None) -> LLMResult:
        generations = []
        gen_kwargs = {}
        self.pipeline.tokenizer.padding_side = "left"
        if self.pipeline.task == "text-generation":
            gen_kwargs = {"return_full_text": False} # not expected by text2text-generation
        prompts_ds = ListDataset(prompts)
        for response in tqdm(self.pipeline(prompts_ds, num_workers=4, **gen_kwargs), total=len(prompts)):
            text = response[0]["generated_text"]
            text = enforce_stop_tokens(text, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

def load_hf_generation_pipeline(
    model_name,
    temperature: float = 0,
    top_p: float = 1.,
    max_tokens: int = 50,
    device: str = "cpu",
    try_optimizations: bool = True,
    generation_kwargs: Optional[dict] = None,
):
    """
    Load a huggingface model, attempting to do optimizations
    """
    gen_model_kwargs = {"device_map": {"": device}}
    if try_optimizations and device == "cuda":
        if is_accelerate_available():
            gen_model_kwargs["device_map"] = "auto"
            if is_bitsandbytes_available():
                gen_model_kwargs["load_in_8bit"] = True
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **gen_model_kwargs)
        task = "text-generation"
    except ValueError:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **gen_model_kwargs)
        task = "text2text-generation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        **generation_kwargs,
    )

    # # if torch version 2 or higher, try to compile model: TODO: does not work currently
    # if torch.__version__.startswith("2") and try_optimizations:
    #     model = torch.compile(model)
    
    # TODO: add generation config to pipeline
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        generation_config=config,
        framework="pt",
        batch_size=16,
    )

    return pipe


def load_adapted_hf_generation_pipeline(
    base_model_name,
    lora_model_name,
    temperature: float = 0,
    top_p: float = 1.,
    max_tokens: int = 50,
    device: str = "cpu",
    try_optimizations: bool = True,
    generation_kwargs: Optional[dict] = None,
):
    """
    Load a huggingface model & adapt with PEFT.
    Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    """
    from peft import PeftModel

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if try_optimizations:
        load_in_8bit = True
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_in_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # does not work currently
    # if torch.__version__ >= "2" and try_optimizations:
    #     model = torch.compile(model)

    generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        **generation_kwargs,
    )
    # TODO: add generation config to pipeline
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        batch_size=8, # TODO: make a parameter
        generation_config=config,
        framework="pt",
    )

    return pipe