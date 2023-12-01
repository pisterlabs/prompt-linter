import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

template = """\
You are a hackthon idea wizard. You provide ideas with real life importance and solving 
problems that are worth solving. Your hacthon subject is {subject}. Provide an idea.
"""

prompt = PromptTemplate.from_template(template)
prompt.format(subject="generative ai")

print(prompt)

llm = OpenAI(openai_api_key="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", model_name='text-davinci-003'
temperature = 0.0)

# llm(prompt.template)

#output parser expected response

print(llm(prompt.template)) 

