"""Chain that calls Google Places API.

Adapting this Google Places API class: https://github.com/hwchase17/langchain/blob/master/langchain/utilities/google_places_api.py
to do what Pasquale defined in the OverpassQuery class in chains_as_classes.py into a similar format to 
"""
import requests
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
    TransformChain,
    LLMChain,
    SimpleSequentialChain,
    SequentialChain,
)
import logging
from langchain.utils import get_from_dict_or_env

from pydantic import BaseModel, Extra, root_validator
from typing import Any, Dict, Optional


class OverpassQueryWrapper:
    """Wrapper around Overpass API.

    The Overpass API is free and requires no setup. Terms of use apply to the servers
    (https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances)

    By default, this will return the all the results on the input query in GeoJSON format.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0, openai_api_key=self.api_key, model_name="gpt-3.5-turbo-0613"
        )
        self.chain_to_overpass_prompt = PromptTemplate(
            input_variables=["user_text_input"],
            template="""Turn the user's message into an overpass QL query.
            Example prompt: "Find bike parking near tech parks in Kreuzberg, Berlin.":\n\n {user_text_input}""",
        )
        self.chain_to_overpass = LLMChain(
            llm=self.llm, prompt=self.chain_to_overpass_prompt, output_key="ql_query"
        )
        self.perform_op_query_chain = TransformChain(
            input_variables=["ql_query"],
            output_variables=["overpass_answer"],
            transform=self.perform_op_query_func,
        )
        self.chain_to_user_prompt = PromptTemplate(
            input_variables=["overpass_answer", "user_text_input"],
            template="""Answer the user's message {user_text_input} based on the result of an overpass QL query contained in {overpass_answer}.""",
        )
        self.chain_to_user = LLMChain(llm=self.llm, prompt=self.chain_to_user_prompt)

        self.overpass_sequential_chain = SequentialChain(
            chains=[
                self.chain_to_overpass,
                self.perform_op_query_chain,
                self.chain_to_user,
            ],
            input_variables=["user_text_input"],
        )

    def overpass_query(self, ql_query: str) -> str:
        """Run an overpass query through the api
        Args:
            ql_query (str): a string with the query in OverpassQL

        Returns:
            str: a json string with the query result.
        """
        overpass_url = "http://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={"data": ql_query})

        if not response.content:
            raise ValueError("Empty response from Overpass API")
        else:
            try:
                data = response.json()
            except:
                raise ValueError(str(response))

        data_str = json.dumps(data)
        return data_str

    def perform_op_query_func(self, inputs: dict) -> dict:
        query_input = inputs["ql_query"]
        op_answer = self.overpass_query(query_input)
        return {"overpass_answer": op_answer}

    def process_user_input(self, user_text_input):
        return self.overpass_sequential_chain.run({"user_text_input": user_text_input})
