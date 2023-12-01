import torch
from langchain import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import LLMChain
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser


model_namemodel = AutoModelForCausalLM.from_pretrained(
        "/storage/ukp/shared/shared_model_weights/models--" + model_name + "/" + model_version,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
    ) = "llama-2-hf"
model_version = "/7B-Chat"
tokenizer = AutoTokenizer.from_pretrained("/storage/ukp/shared/shared_model_weights/models--"+model_name+"/"+model_version)
model = AutoModelForCausalLM.from_pretrained(
        "/storage/ukp/shared/shared_model_weights/models--" + model_name + "/" + model_version,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
    )