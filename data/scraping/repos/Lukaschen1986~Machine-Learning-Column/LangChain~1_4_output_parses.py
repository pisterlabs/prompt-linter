# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/model_io/output_parsers/
https://www.langchain.com.cn/modules/prompts/output_parsers
"""
import os
import torch as th
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          T5Tokenizer, T5ForConditionalGeneration, pipeline)
from langchain.llms import OpenAI
from langchain import (HuggingFaceHub, PromptTemplate, LLMChain)
from langchain.llms import HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser


