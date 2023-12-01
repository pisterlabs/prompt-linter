import requests
import os
import sys
import json

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI,  LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re


URL = "https://api.subquery.network/sq/baidang201/chatpolkadot"

import logging
# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


JUDGEMENT_PROMPT = '''
tell me the total xcm extrinsics count
'''

from core.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  

import logging
# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_query_params(x):
    response_schemas = [
      ResponseSchema(name="question", description="question is the problem itself.for example,'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B.',would be 'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B.'"),
      ResponseSchema(name="params", description="The parameter extracted from the question, for instance 'can you check the balance of this address 0x60e4d786628fea6478f785a6d7e704777c86a7c6?', would be '0x60e4d786628fea6478f785a6d7e704777c86a7c6'.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=JUDGEMENT_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM

    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    print("INFO:     Judge Result:", result)

    return result


# Headers
HEADERS = {
    "Content-Type": "application/json",
}


def fetch_xcm_txs_count(question):
    data = get_query_params(question)

    TOKEN_TRANSFERS = f''' 
    query MyQuery {{
        extrinsics(
        filter: {{
          module: {{ equalTo: "xcmPallet" }}
        }}
      ) {{
        totalCount
      }}
    }}
    '''

    # Make the request
    response = requests.post(URL, headers=HEADERS, json={'query': TOKEN_TRANSFERS})

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=4))  # print the data
        return data
    else:
        return {"error": "Query failed", "status_code": response.status_code}
