# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/model_io/prompts/
https://www.langchain.com.cn/modules/prompts

编程模型的新方式是通过提示进行的。 “提示”指的是模型的输入。 这个输入很少是硬编码的，而是通常从多个组件构建而成的。 
PromptTemplate负责构建这个输入。 LangChain提供了几个类和函数，使构建和处理提示变得容易。
"""
from langchain import (PromptTemplate, FewShotPromptTemplate)
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import load_prompt


# ----------------------------------------------------------------------------------------------------------------
# 单个输入
template = (
    "I want you to act as a naming consultant for new companies. "
    "What is a good name for a company that makes {product}? "
    )
prompt = PromptTemplate.from_template(template)
prompt.format(product="colorful socks")

# ----------------------------------------------------------------------------------------------------------------
# 多个输入
template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate.from_template(template)
prompt.format(adjective="funny", content="chickens")
'''
Tell me a funny joke about chickens.
'''

# ----------------------------------------------------------------------------------------------------------------
# FewShotPromptTemplate
template = (
    "Word: {word}  "
    "Antonym: {antonym}\n"
    )
prompt = PromptTemplate.from_template(template)

few_shot_examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"}
]

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=prompt,
    prefix="Give the antonym of every input:\n",
    suffix="Word: {input}  Antonym:",
    input_variables=["input"],
    example_separator="-> "
    )
print(few_shot_prompt.format(input="big"))

# ----------------------------------------------------------------------------------------------------------------
# FewShotPromptTemplate + LengthBasedExampleSelector
template = (
    "Word: {word}  "
    "Antonym: {antonym}\n"
    )
prompt = PromptTemplate.from_template(template)

few_shot_examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"}
]

example_selector = LengthBasedExampleSelector(
    examples=few_shot_examples,
    example_prompt=prompt,
    max_length=25
    )

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt,
    prefix="Give the antonym of every input:\n",
    suffix="Word: {input}  Antonym:",
    input_variables=["input"],
    example_separator="-> "
    )

print(dynamic_prompt.format(input="big"))

# ----------------------------------------------------------------------------------------------------------------
# 模板序列化
prompt.save(file_path="awesome_prompt.json")
prompt = load_prompt(path="awesome_prompt.json")
 

