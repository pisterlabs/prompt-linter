# package import
import numpy as np
import pandas as pd
import matplotlib 
import plotly
from transformers import AutoTokenizer
import transformers
import torch
import langchain
from langchain import HuggingFacePipeline, LLMChain

from parameters import LLM_model 


# LLM (Llamma 2) set up
model = LLM_model
tokenizer = AutoTokenizer.from_pretrained(LLM_model)

pipeline = transformers.pipeline(
    "text-generation", 
    model=LLM_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    eos_token_id=tokenizer.eos_token_id
)

print("pipeline setup finished")

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

print("llm setup finished")

template = """
              You are an intelligent chatbot that gives out useful information to humans.
              You return the responses in sentences with arrows at the start of each sentence
              {query}
           """

prompt = langchain.prompts.PromptTemplate(template=template, input_variables=["query"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run('What are the 3 causes of glacier meltdowns?'))





