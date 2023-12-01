import json
from pprint import pprint
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import requests
from config import COHERE_MODEL, VECTOR_NAME, client, co
from termcolor import colored
from tqdm import tqdm


def list_all_indices() -> Dict.keys:
    """Helper function to list all indices in opensearch"""
    return client.indices.get_alias("*").keys()


def delete_index(index_name: str) -> None:
    """Helper function to delete an index in opensearch"""
    response = client.indices.delete(index=index_name)
    return pprint(response)


def get_schema_mapping(index_name: str) -> None:
    """Retrieve schema mapping for the index_name specified.
    The mapping lists all the fields and their data types."""

    mapping_data = client.indices.get_mapping(index_name)

    # find index doc_type
    doc_type = list(mapping_data[index_name]["mappings"].keys())[0]

    schema = mapping_data[index_name]["mappings"][doc_type]

    pprint(list(schema.keys()))
    print("\n")
    pprint(schema, width=80, indent=4)


def batch_embed(texts: List[str], batch_size: int = 256) -> List[np.array]:
    """
    Embed a list of text with cohere embedding model.
    """
    embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size)):
        embeddings.extend(
            np.array(
                co.embed(
                    texts[start_idx : start_idx + batch_size],
                    model=COHERE_MODEL,
                ).embeddings
            )
        )
    return embeddings


def get_cohere_embedding(
    text: Union[str, List[str]], model_name: str = "embed-english-light-v2.0"
) -> List[float]:
    """
    Embed a single text with cohere client and return list of floats
    """
    if type(text) == str:
        embed = co.embed([text], model=model_name).embeddings[0]
    else:
        embed = co.embed(text, model=model_name).embeddings
    return embed


def search_match_phrase(field: str, query: str, index_name: str) -> Dict:
    """
    Search by match phrase for specific phrases in a field.
    """
    assert index_name in set(list_all_indices()), f"{index_name} not created"
    print(f"Searching for `{query}` in the field `{field}`")
    query_body = {"query": {"match_phrase": {field: {"query": query}}}}
    response = client.search(index=index_name, body=query_body)
    return response


def search_fuzzy(field: str, query: str, fuzziness: int, index_name: str) -> Dict:
    """
    Search by specifying fuzziness to account for typos and misspelling.
    """
    print(
        f"Search for `{query}` in the `{field}` field with fuzziness set to {fuzziness}"
    )
    assert index_name in set(list_all_indices()), f"{index_name} not created"

    query_body = {
        "query": {
            "fuzzy": {
                field: {
                    "value": query,
                    "fuzziness": fuzziness,
                }
            }
        }
    }
    response = client.search(index=index_name, body=query_body)
    return response


def find_similar_docs(query: str, k: int, num_results: int, index_name: str) -> Dict:
    """
    Main vector search capability using knn on input query strings.
    Args:
        k: number of k-similar neighbors/vectors to retrieve from opensearch index
        num_results: number of the k-similar vectors to retrieve.
        index_name: index name in opensearch
    """
    embed_vector = get_cohere_embedding(query)

    body = {
        "size": num_results,
        "query": {"knn": {VECTOR_NAME: {"vector": embed_vector, "k": k}}},
    }

    url = f"http://localhost:9200/{index_name}/_search"
    response = requests.get(
        url, json=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(response.content)


def format_search_output(out: Dict) -> pd.DataFrame:
    """
    Helper function to format output from _search endpoints that
    return list of lists and need to be formatted into a dataframe
    """
    columns = ["score", "abstract", "title", "arxiv_id", "embeddings"]
    try:
        out = out["hits"]["hits"]
        data = [
            [
                o["_score"],
                o["_source"]["text"],
                o["_source"]["title"],
                o["_source"]["arxiv_id"],
                o["_source"][VECTOR_NAME],
            ]
            for o in out
        ]
        
        df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f"couldnt format output because of {e}")
        df = pd.DataFrame([], columns=columns)
    return df.sort_values(by="score", ascending=False)


def colorize(sentence: str, words: Union[List[str], str], color: str = "blue") -> str:
    """Visualization function that will highlight the query words
    in a sentence with the color provided"""
    sentence = sentence.lower()
    if type(words) == list:
        for word in words:
            sentence = sentence.replace(
                word.lower(), colored(word, color, attrs=["reverse", "blink"])
            )
    elif type(words) == str:
        sentence = sentence.replace(
            words.lower(), colored(words, color, attrs=["reverse", "blink"])
        )
    else:
        raise f"cannot colorize {sentence}"
    return sentence


def colorize_st(
    sentence: str, words: Union[List[str], str], color: str = "blue"
) -> str:
    """Visualization function that will highlight the query words
    in a sentence with the color provided"""
    sentence = sentence.lower()
    if type(words) == list:
        for word in words:
            sentence = sentence.replace(word.lower(), f"**:{color}[{word.lower()}]**")
    elif type(words) == str:
        sentence = sentence.replace(words.lower(), f"**:{color}[{words.lower()}]**")
    else:
        raise f"cannot colorize {sentence}"
    return sentence
