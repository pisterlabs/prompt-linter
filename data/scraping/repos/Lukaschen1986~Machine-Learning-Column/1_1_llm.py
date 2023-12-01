# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/model_io/models/llms/
https://www.langchain.com.cn/modules/models/llms/getting_started

LLM类是设计用于与LLMs进行接口交互的类。有许多LLM提供商（OpenAI、Cohere、Hugging Face等)
该类旨在为所有LLM提供商提供标准接口。在本文档的这部分中，我们将重点介绍通用LLM功能。
有关使用特定LLM包装器的详细信息，请参见如何指南中的示例。
"""
import os
import torch as th
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          T5Tokenizer, T5ForConditionalGeneration, pipeline)
from langchain.llms import OpenAI
from langchain import (HuggingFaceHub, PromptTemplate, LLMChain)
from langchain.llms import HuggingFacePipeline


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 1-使用 OpenAI 模型
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
llm(prompt="hello")
'''
Retrying langchain.llms.openai.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds 
as it raised RateLimitError: You exceeded your current quota, please check your plan and billing details..
'''

# ----------------------------------------------------------------------------------------------------------------
# 2-使用本地 HuggingFace 模型（根据不同模型的要求定义加载方法）
checkpoint = "flan-t5-large"  # https://huggingface.co/google/flan-t5-large#usage
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True
)
pretrained = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=False,
    load_in_8bit=False,
    device_map="auto"
    )  # GPU

# checkpoint = "chatglm3-6b"
# tokenizer = AutoTokenizer.from_pretrained(
#     pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
#     cache_dir=path_model,
#     force_download=False,
#     local_files_only=True,
#     trust_remote_code=True
#     )
# pretrained = AutoModel.from_pretrained(
#     pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
#     cache_dir=path_model,
#     force_download=False,
#     local_files_only=True,
#     trust_remote_code=True
#     ).quantize(8).cuda()
# model = pretrained.eval()
# model.chat(tokenizer, "你好", history=[])

# ----------------------------------------------------------------------------------------------------------------
# text2text-generation
# text-generation
pipe = pipeline(
    task="text2text-generation",
    tokenizer=tokenizer,
    model=pretrained,
    max_length=100
    )

llm = HuggingFacePipeline(pipeline=pipe)

res = llm(prompt="What is the capital of China?")  # beijing
res = llm(prompt="Tell me a joke")  # if you want to be a sailor, you have to be a sailor.
type(res)  # str

res = llm.generate(prompts=["Tell me a joke", "Tell me a poem"]*2)
type(res)  # langchain.schema.output.LLMResult
type(res.generations)  # list
len(res.generations)  # 4

res.generations[0]  # [Generation(text='if you want to be a sailor, you have to be a sailor.')]
res.llm_output

llm.get_num_tokens("what a joke")  # 3

# ----------------------------------------------------------------------------------------------------------------
# 3-集成 PromptTemplate 和 LLMChain
template = (
    "Question: {question}\n"
    "Answer: Let's think step by step."  # "Answer: Give me the answer directly."
    )

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["question"]
#     )
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=prompt)

question = "What is the capital of China?"
llm_chain.run(question)
'''
The capital of China is Beijing. Beijing is the capital of China. So the answer is Beijing.
'''






