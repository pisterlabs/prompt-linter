# -*- coding: utf-8 -*-
"""Copy of YT LangChain RAG tips and Tricks 01 - Self Query.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zDYbCfGVMs4pHmqlmOtLr3ukCSc1kfXD
"""
import os
import constants
import together
import logging
from typing import Any, Dict, List, Mapping, Optional
from langchain.llms import OpenAI

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from prettytable import PrettyTable
from termcolor import colored

os.environ["TOGETHER_API_KEY"] = constants.TOGETHER_API_KEY
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# set your API key
together.api_key = os.environ["TOGETHER_API_KEY"]

# print the first model's name
# models = together.Models.list()
# print(models[3]["name"]), print(models[52]["name"])
# for idx, model in enumerate(models):
#     print(idx, model["name"])

# print(models[55]["name"])

# together.Models.start("mistralai/Mistral-7B-Instruct-v0.1")


class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "mistralai/Mistral-7B-v0.1"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.0
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    # class Config:
    #     extra = 'forbid'

    # @root_validator(skip_on_failure=True)
    # def validate_environment(cls, values: Dict) -> Dict:
    #     """Validate that the API key is set."""
    #     api_key = get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
    #     values["together_api_key"] = api_key
    #     return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = output["output"]["choices"][0]["text"]
        return text


# llm = TogetherLLM(
#     model="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.1, max_tokens=512
# )

# type(llm), llm.model, llm.temperature

# print("Q: What are the olympics? ")
# print(llm("What are the olympics? "))

# """## Self-querying Retriever"""

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

"""## Formatting and printing results"""


def print_documents(docs):
    table = PrettyTable()
    table.field_names = [
        "Page Content",
        "Color",
        "Country",
        "Grape",
        "Name",
        "Rating",
        "Year",
    ]

    for doc in docs:
        table.add_row(
            [
                doc.page_content,
                colored(doc.metadata["color"], "red"),
                colored(doc.metadata["country"], "yellow"),
                colored(doc.metadata["grape"], "blue"),
                colored(doc.metadata["name"], "green"),
                colored(doc.metadata["rating"], "magenta"),
                colored(doc.metadata["year"], "cyan"),
            ]
        )
    print(table)


"""## Example data with metadata attached"""

docs = [
    Document(
        page_content="Complex, layered, rich red with dark fruit flavors",
        metadata={
            "name": "Opus One",
            "year": 2018,
            "rating": 96,
            "grape": "Cabernet Sauvignon",
            "color": "red",
            "country": "USA",
        },
    ),
    Document(
        page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
        metadata={
            "name": "Château d'Yquem",
            "year": 2015,
            "rating": 98,
            "grape": "Sémillon",
            "color": "white",
            "country": "France",
        },
    ),
    Document(
        page_content="Full-bodied red with notes of black fruit and spice",
        metadata={
            "name": "Penfolds Grange",
            "year": 2017,
            "rating": 97,
            "grape": "Shiraz",
            "color": "red",
            "country": "Australia",
        },
    ),
    Document(
        page_content="Elegant, balanced red with herbal and berry nuances",
        metadata={
            "name": "Sassicaia",
            "year": 2016,
            "rating": 95,
            "grape": "Cabernet Franc",
            "color": "red",
            "country": "Italy",
        },
    ),
    Document(
        page_content="Highly sought-after Pinot Noir with red fruit and earthy notes",
        metadata={
            "name": "Domaine de la Romanée-Conti",
            "year": 2018,
            "rating": 100,
            "grape": "Pinot Noir",
            "color": "red",
            "country": "France",
        },
    ),
    Document(
        page_content="Crisp white with tropical fruit and citrus flavors",
        metadata={
            "name": "Cloudy Bay",
            "year": 2021,
            "rating": 92,
            "grape": "Sauvignon Blanc",
            "color": "white",
            "country": "New Zealand",
        },
    ),
    Document(
        page_content="Rich, complex Champagne with notes of brioche and citrus",
        metadata={
            "name": "Krug Grande Cuvée",
            "year": 2010,
            "rating": 93,
            "grape": "Chardonnay blend",
            "color": "sparkling",
            "country": "New Zealand",
        },
    ),
    Document(
        page_content="Intense, dark fruit flavors with hints of chocolate",
        metadata={
            "name": "Caymus Special Selection",
            "year": 2018,
            "rating": 96,
            "grape": "Cabernet Sauvignon",
            "color": "red",
            "country": "USA",
        },
    ),
    Document(
        page_content="Exotic, aromatic white with stone fruit and floral notes",
        metadata={
            "name": "Jermann Vintage Tunina",
            "year": 2020,
            "rating": 91,
            "grape": "Sauvignon Blanc blend",
            "color": "white",
            "country": "Italy",
        },
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)
print("vectorstore", vectorstore)

"""## Creating our self-querying retriever"""

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="grape",
        description="The grape used to make the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="name",
        description="The name of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="color",
        description="The color of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the wine was released",
        type="integer",
    ),
    AttributeInfo(
        name="country",
        description="The name of the country the wine comes from",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The Robert Parker rating for the wine 0-100",
        type="integer",  # float
    ),
]
document_content_description = "Brief description of the wine"

# Assuming 'document_contents' is a list of the content of each document
document_contents = [doc.page_content for doc in docs]

llm_openai = OpenAI(temperature=0)
from langchain.llms import VertexAI
llm_google = VertexAI()

retriever = SelfQueryRetriever.from_llm(
    llm_google,
    #llm_openai,  # THIS WORKS
    # llm,  # THIS DOES NOT WORK, reason according to Sam Witteveen "you will need a model that can handle JSON output well. I suggest trying some of the code models. If I am using an opensource model for this kind of task I will often fine tune it for the application first. Hope that helps".
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)
# This example only specifies a relevant query
print("Q: What are some red wines")
print_documents(retriever.get_relevant_documents("What are some red wines"))

print("Q: Who is Gary Oldman? ")
print(llm_google("Who is Gary Oldman? "))
