# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import openai
from dotenv import load_dotenv
import json
import pandas as pd
import inspect
import re

from langchain import OpenAI, PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain.callbacks import get_openai_callback
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools
# For Azure GPT keys
from langchain.chat_models import AzureChatOpenAI
# For non-Azure keys
from langchain.llms import OpenAI

# Load environ variables from .env, will not override existing environ variables
load_dotenv()

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# For GPT in Azure
llm = AzureChatOpenAI(
    openai_api_type='azure',
    openai_api_base=OPENAI_API_BASE,
    openai_api_version="2023-05-15",
    deployment_name='gpt-4-32k',
    model_name='gpt-4-32k',
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=4000,
    )

# Without Azure key
# llm = OpenAI(
#     model_name='text-davinci-003',
#     temperature=0,
#     max_tokens=2048,
#     openai_api_key=OPENAI_API_KEY
#     )


prefix = """
Generate the answer of graph processing queries. 
User will specify the output type. The return_object will be a JSON object with two keys, 'type' and 'data'. 
The 'type' key should indicate the output format depending on the user query or request. It should be one of 'list', 'table' or 'graph'.
The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should contain the updated graph JSON.

Remember, your reply should always start with string "Answer:\n'''".
"""

suffix = """ Please answer the following question:

Answer:
'''
"type": list or table or graph
"data": LLM output list or table or graph
'''

Question: {input}
"""

prompt = PromptTemplate(
    input_variables=["input"],
    template=prefix+suffix
)


pyGraphNetExplorer = LLMChain(llm=llm, prompt=prompt)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return cb.total_tokens


llm_input_token_count = count_tokens(pyGraphNetExplorer, query=prompt)

