import os
import constants
import together
from typing import Any, Dict, Optional
from pydantic import Extra, Field, root_validator
from langchain.llms.base import LLM
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from prettytable import PrettyTable
from termcolor import colored
import langchain

langchain.verbose = True
langchain.debug = True
os.environ["TOGETHER_API_KEY"] = constants.TOGETHER_API_KEY
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY


class TogetherLLM(LLM):
    """Large language models from Together."""

    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    temperature: float = 0.0
    max_tokens: int = 2600

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = os.environ["TOGETHER_API_KEY"]
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        print("\n\nRAG-output:", output)
        text = output["output"]["choices"][0]["text"]
        print("\n\nRAG-text before cleanup:", text)

        # substrings to be removed from start and end
        beginning_str = "```json\n"
        end_str = "\n```"

        # removing beginning_str and end_str from the text
        if text.startswith(beginning_str):
            text = text[len(beginning_str) :]

        if text.endswith(end_str):
            text = text[: -len(end_str)]

        print("\n\nRAG-text after cleanup:", text)
        return text


llm = TogetherLLM()
llm_openai = OpenAI(temperature=0)

# embeddings = HuggingFaceInstructEmbeddings(
#     model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
# )
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
)

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
        type="integer",
    ),
]

document_content_description = "Brief description of the wine"


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


print("Q: Who is Gary Oldman? ")
print(llm("Who is Gary Oldman? "))

retriever = SelfQueryRetriever.from_llm(
    llm_openai,  # THIS WORKS
    # llm,  # THIS DOES NOT WORK, reason according to Sam Witteveen "you will need a model that can handle JSON output well. I suggest trying some of the code models. If I am using an opensource model for this kind of task I will often fine tune it for the application first. Hope that helps".
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)
# Currently only openAi (llm_openai) works with SelfQueryRetriever. Why???
# It appears the error originates from the output of the TogetherLLM model. After running the input query, it returns a JSON object which is then parsed. Based on the error message, the JSON object that has been returned by the model has some extra data that's causing the issue.

# The traceback shows that the error occurs during execution of the parse_and_check_json_markdown function in json.py:

# sql
# Copy code
# json.decoder.JSONDecodeError: Extra data: line 5 column 1 (char 68)
# The error "Extra Data" typically occurs when there is extra data outside of the structure of a JSON object, such as multiple JSON objects not encapsulated within a JSON array. Seeing from the error message, it seems there are multiple JSON objects being returned by Together's LLM. And if the parser expects only ONE object, it's failing when encounters the start of the next object.

# A potential workaround is to modify the TogetherLLM to ensure that it returns single, well-formatted JSON text that the rest of your code can handle.

# Another approach would be to extend the parser's functionality to process multiple JSON objects.

# However, the best solution would depend on the exact specifications of your project and the outputs that Together's LLM is supposed to return for successful integration with the SelfQueryRetriever. If the LLM model often returns multiple separate JSON objects instead of just one, you may wish to consider adjusting the parser accordingly. But, if this is not expected behavior for the LLM, then adjusting the model to only return a single JSON object may be more efficient.

# Lastly, it is always recommended to reach out to the library owners or maintainers for assistance with such issues. They may have more specific insights into why such a problem might occur and how best to resolve it.

print("Q: What are some red wines")
print_documents(retriever.get_relevant_documents("What are some red wines"))

# print("Q: I want a wine that has fruity nodes")
# print_documents(retriever.get_relevant_documents("I want a wine that has fruity nodes"))

# # This example specifies a query and a filter
# print("Q: I want a wine that has fruity nodes and has a rating above 97")
# print_documents(
#     retriever.get_relevant_documents(
#         "I want a wine that has fruity nodes and has a rating above 97"
#     )
# )

# print("Q: What wines come from Italy?")
# print_documents(retriever.get_relevant_documents("What wines come from Italy?"))

# # This example specifies a query and composite filter
# print("Q: What's a wine after 2015 but before 2020 that's all earthy")
# print_documents(
#     retriever.get_relevant_documents(
#         "What's a wine after 2015 but before 2020 that's all earthy"
#     )
# )

# """## Filter K

# We can also use the self query retriever to specify k: the number of documents to fetch.

# We can do this by passing enable_limit=True to the constructor.
# """

# retriever = SelfQueryRetriever.from_llm(
#     llm,
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
