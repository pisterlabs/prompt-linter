# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Module for MSTICPy natural language function search."""
import contextlib
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
from langchain.document_loaders import DataframeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

from msticpy.datamodel import entities
from msticpy.data.core.data_providers import QueryProvider
from msticpy.init.pivot_core.pivot_container import PivotContainer


from .retrieval_base import RetrievalBase
from .vector_stores import create_vector_store, read_vector_store
from ._version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

1. Find the most relevant function that could answer the Question
2. Suggest up to 3 additional functions that might be relevant to the question
Format the answer as a JSON object with the following fields:
- "function": the name of the function
- "additional_functions": a list of up to 5 additional functions that might be relevant to the question

{context}

Question: {question}

"""

MP_VS_PATH = "E:/src/chat-langchain/mp-rtd-vs.pkl"


class RTDocSearch(RetrievalBase):
    """Ask questions about the MSTICPy Documentation."""

    def __init__(
        self,
        vs_path: str = MP_VS_PATH,
        max_tokens: int = 1000,
        memory: bool = False,
        verbose: bool = False,
    ):
        """Initialize the code search."""
        super().__init__(vs_path or MP_VS_PATH, max_tokens, memory, verbose)

    @classmethod
    def create_vectorstore(cls, qry_provs: List[QueryProvider], vs_path: str):
        """
        Read in MP functions to vector store and save.

        Parameters
        ----------
        qry_provs : List[QueryProvider]
            Path to read HTML documents from.
        vs_path : str
            Path to save pickled vectorstore to.

        """
        logger.info("Loading data.")
        if Path(vs_path).exists():
            with contextlib.suppress(ValueError):
                return read_vector_store(vs_path, f"{cls.__name__}.create_vector_store")

        comb_funcs_df = enumerate_mp_functions(qry_provs)
        return func_df_to_vectorstore(comb_funcs_df, "search_text", vs_path)


def search_doc(question: str):
    """Search the code for the answer to the question."""
    response = mp_doc_search.ask(question)
    return Markdown(response)


# ---------------------------------------------------------------------
# Read MSTICPY functions
# ---------------------------------------------------------------------


# Functions to retrieve MSTICPy functions
# ---------------------------------------


def get_entity_classes():
    """Return a dictionary of entity names and classes."""
    entity_classes = {}
    for item_name in dir(entities):
        item = getattr(entities, item_name)
        if isinstance(item, type) and issubclass(item, entities.Entity):
            entity_classes[item.__name__] = item
    return entity_classes


def get_pivot_func(entity, func_name):
    """Get the pivot function for an entity and pivot name."""
    func_parts = func_name.split(".")
    curr_node = entity
    for func_part in func_parts:
        curr_node = getattr(curr_node, func_part)
        if isinstance(curr_node, PivotContainer):
            continue
        elif callable(curr_node):
            return {
                "func": curr_node,
                "name": func_name,
                "description": curr_node.__doc__.strip().split("\n")[0],
                "docstring": curr_node.__doc__.strip(),
            }
    print(func_name, func_parts, curr_node, entity)
    raise TypeError("no pivot function found")


def get_pivot_functions():
    """Return a dataframe of pivot functions."""
    entity_classes = get_entity_classes()
    entity_pivot_funcs = {}
    for entity in entity_classes.values():
        pivot_dict = {pivot: get_pivot_func(entity, pivot) for pivot in entity.pivots()}
        if pivot_dict:
            entity_pivot_funcs[entity.__name__] = pivot_dict

    pivot_list = []
    for entity in entity_classes.values():
        for pivot_name in entity.pivots():
            pivot_props = get_pivot_func(entity, pivot_name)
            pivot_list.append(
                {
                    "type": "entity_pivot_function",
                    "function_name": f"{entity.__name__}.{pivot_name}",
                    "example": f'{entity.__name__}.{pivot_name}("<value>"|<list of values>)',
                    "entity": entity.__name__,
                    "description": pivot_props["description"],
                    "docstring": pivot_props["docstring"],
                }
            )

    return pd.DataFrame(pivot_list)


def get_pandas_ext_methods():
    """Return a dataframe of pandas extension methods."""
    pd_ext_methods = []
    for attrib in dir(df):
        if attrib.startswith("mp"):
            ext_attrib = getattr(df, attrib)
            for ext_func_name in dir(ext_attrib):
                if not ext_func_name.startswith("_"):
                    doc = getattr(ext_attrib, ext_func_name).__doc__
                    pd_ext_methods.append(
                        {
                            "type": "pandas_extension",
                            "function_name": f"df.{attrib}.{ext_func_name}",
                            "example": f"df.{attrib}.{ext_func_name}(<args...)",
                            "description": doc.strip().split("\n")[0],
                            "docstring": doc,
                        }
                    )

    return pd.DataFrame(pd_ext_methods)


_GENERIC_QUERY_PARAMS = [
    "start",
    "end",
    "add_query_items",
    "time_column",
    "query_project",
    "table",
]


def get_query_functions(qry_prov):
    """Return a dataframe of query functions."""
    query_desc = [
        {
            "type": "data_query",
            "function_name": f"{query}",
            "example": f"qry_prov.{query}(start=<start>, end=<end>, <parameters...>)",
            "description": qry_prov.query_store[query].description,
            "docstring": qry_prov.query_store[query].create_doc_string(),
            "parameters": ", ".join(
                [
                    param
                    for param in qry_prov.query_store[query].params
                    if param not in _GENERIC_QUERY_PARAMS
                ]
            ),
            "additional_parameters": ", ".join(
                [
                    param
                    for param in qry_prov.query_store[query].params
                    if param in _GENERIC_QUERY_PARAMS
                ]
            ),
        }
        for query in qry_prov.list_queries()
    ]
    return pd.DataFrame(query_desc)


# ## Combine dataframes


def enumerate_mp_functions(qry_providers: Optional[List[QueryProvider]]):
    """Return a dataframe of MSTICPy functions."""
    pivot_df = get_pivot_functions()
    pd_ext_methods_df = get_pandas_ext_methods()

    query_func_list = [get_query_functions(qry_prov) for qry_prov in qry_providers]
    query_desc_df = pd.concat(query_func_list, ignore_index=True)
    return combine_func_dfs([pd_ext_methods_df, query_desc_df, pivot_df])


def combine_func_dfs(df_list):
    """Return combined func dataframe with 'search_text' column."""
    comb_df = pd.concat(df_list, ignore_index=True)
    comb_df.head()
    comb_df["search_text"] = (
        "function_name: "
        + comb_df["function_name"]
        + ", type: "
        + comb_df["type"]
        + ", description: "
        + comb_df["description"]
        + ", docstring: "
        + comb_df["docstring"]
    )
    return comb_df


comb_funcs_df = enumerate_mp_functions([qry_prov])

# ----------------------------------------------
# Create OpenAI embeddings
# ----------------------------------------------


def func_df_to_vectorstore(data, column, vs_path: str = "mp_funcs_vectorstore.pkl"):
    """Get documents from web pages."""
    loader = DataFrameLoader(data, page_content_column=column)
    raw_documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    # )
    # documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(raw_documents, embeddings)

    # Save vectorstore
    with open(vs_path, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore


# ----------------------------------------------
# Format output
# ----------------------------------------------


def find_mp_function(question):
    """Find the most relevant function that could answer the question."""
    response = mp_func_prompt(question)
    answer = response["result"]
    if answer == "I don't know":
        Markdown("Could not find an answer to the question.")
    if answer.startswith("Answer:"):
        answer = answer.replace("Answer:", "").strip()
    answer_dict = json.loads(answer)
    main_function = answer_dict["function"]
    additional_functions = answer_dict["additional_functions"]
    return display_mp_func_details(main_function, additional_functions)


def display_mp_func_details(
    function_name: str, additional_functions: Optional[List[str]] = None
):
    """Return the details of the function as Markdown."""
    data_match = comb_funcs_df[comb_funcs_df["function_name"] == function_name].iloc[0]
    return format_response(data_match, additional_functions)


_RESPONSE_FORMAT = """
### Function name: `{function}`

Function type: **{type}**

Description: **{description}**

Example:
```python
{example}
```

#### Other possible matches:
{other_matches}

### Function docstring:

```
{docstring}
```

"""


def format_response(data_match, other_matches: Optional[List[str]] = None):
    """Format markdown response to question."""
    if other_matches:
        other_funcs = "\n".join([f"- {match}" for match in other_matches])
    else:
        other_funcs = "None"

    return Markdown(
        _RESPONSE_FORMAT.format(
            function=data_match["function_name"],
            type=data_match["type"],
            description=data_match["description"],
            example=data_match["example"],
            docstring=data_match["docstring"],
            other_matches=other_funcs,
        )
    )
