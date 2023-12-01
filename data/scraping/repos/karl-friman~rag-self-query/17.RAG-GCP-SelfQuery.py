import os
import constants
import logging
from typing import Any, Dict, List, Mapping, Optional
from langchain.llms import OpenAI

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import VertexAI

from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from prettytable import PrettyTable
from termcolor import colored

#ToDO: replace with VertexAI
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings

#ToDO: replace with GCP solution, pgvector?
from langchain.vectorstores import Chroma

import langchain

#ToDO: turn these 2 off as a final stage
langchain.verbose = True
langchain.debug = True

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

llm_google = VertexAI()

retriever = SelfQueryRetriever.from_llm(
    llm_google,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)
# This example only specifies a relevant query
print("Q: What are some red wines")
print_documents(retriever.get_relevant_documents("What are some red wines"))

print("Q: I want a wine that has fruity nodes")
print_documents(retriever.get_relevant_documents("I want a wine that has fruity nodes"))

# This example specifies a query and a filter
print("Q: I want a wine that has fruity nodes and has a rating above 97")
print_documents(
    retriever.get_relevant_documents(
        "I want a wine that has fruity nodes and has a rating above 97"
    )
)

print("Q: What wines come from Italy?")
print_documents(retriever.get_relevant_documents("What wines come from Italy?"))

# This example specifies a query and composite filter
print("Q: What's a wine after 2015 but before 2020 that's all earthy")
print_documents(
    retriever.get_relevant_documents(
        "What's a wine after 2015 but before 2020 that's all earthy"
    )
)

#ToDO: This used to work but doesn't anymore, there may have been a change to how the API works

# """## Filter K

# We can also use the self query retriever to specify k: the number of documents to fetch.

# We can do this by passing enable_limit=True to the constructor.
# """
# llm_openai = OpenAI(temperature=0)
# retriever = SelfQueryRetriever.from_llm(
#     llm_openai,
#     #llm_google, #This does not work
#     vectorstore,
#     document_content_description,
#     metadata_field_info,
#     enable_limit=True,
#     verbose=True,
# )
# print("Q: what are two that have a rating above 97")
# # This example only specifies a relevant query - k= 2
# print_documents(
#     retriever.get_relevant_documents("what are two that have a rating above 97")
# )
# print("Q: what are two wines that come from australia or New zealand")
# print_documents(
#     retriever.get_relevant_documents(
#         "what are two wines that come from australia or New zealand"
#     )
# )
