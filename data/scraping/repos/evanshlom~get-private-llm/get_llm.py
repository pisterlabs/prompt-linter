from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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' # no longer active ;)

model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=False)
model.save_pretrained('flan-t5-large')

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
