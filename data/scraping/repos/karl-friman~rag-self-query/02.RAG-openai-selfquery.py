# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# Imports
import os
import constants
from prettytable import PrettyTable
from termcolor import colored
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Set API Key from constants
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()


def print_documents(docs):
    """Formats and prints documents in a table format."""
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

# Metadata field info
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
llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

# Sample Queries and Display
print("What are some red wines")
print_documents(retriever.get_relevant_documents("What are some red wines"))

print("I want a wine that has fruity nodes")
print_documents(retriever.get_relevant_documents("I want a wine that has fruity nodes"))

print("I want a wine that has fruity nodes and has a rating above 97")
print_documents(
    retriever.get_relevant_documents(
        "I want a wine that has fruity nodes and has a rating above 97"
    )
)

print("What wines come from Italy?")
print_documents(retriever.get_relevant_documents("What wines come from Italy?"))

print("What's a wine after 2015 but before 2020 that's all earthy")
print_documents(
    retriever.get_relevant_documents(
        "What's a wine after 2015 but before 2020 that's all earthy"
    )
)

# Filter K - Using the self-query retriever to specify k: number of documents to fetch
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

print("what are two that have a rating above 97")
print_documents(
    retriever.get_relevant_documents("what are two that have a rating above 97")
)
print("what are two wines that come from australia or New zealand")
print_documents(
    retriever.get_relevant_documents(
        "what are two wines that come from australia or New zealand"
    )
)
