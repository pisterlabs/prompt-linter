import os
import cohere
from typing import List
from qdrant_client import QdrantClient
from qdrant_client import models

from .constants import (
    MULTILINGUAL_EMBEDDING_MODEL,
    ENGLISH_EMBEDDING_MODEL,
    SEARCH_QDRANT_COLLECTION_NAME,
    TRANSLATE_BASED_ON_USER_QUERY,
    TEXT_GENERATION_MODEL,
    USE_MULTILINGUAL_EMBEDDING,
)

# load environment variables
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# create qdrant and cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
)


def embed_user_query(user_query: str) -> List:
    """
    Create an embedding for the given query by the user using Cohere's Embed API.

    Args:
        user_query (`str`):
            The input query by the user based on which search will be performed with the help of Qdrant.

    Returns:
        query_embedding (`List`):
            A list of numbers or vector representing the user query.
    """
    if USE_MULTILINGUAL_EMBEDDING:
        model_name = MULTILINGUAL_EMBEDDING_MODEL
    else:
        model_name = ENGLISH_EMBEDDING_MODEL

    embeddings = cohere_client.embed(
        texts=[user_query],
        model=model_name,
    )
    query_embedding = embeddings.embeddings[0]
    return query_embedding


def search_docs_for_query(
    query_embedding: List,
    num_results: int,
    user_query: str,
    languages: List,
    match_text: List,
) -> List:
    """
    Perform search on the collection of documents for the given user query using Qdrant's search API.
        Args:
            query_embedding (`List`):
                A vector representing the user query.
            num_results (`str`):
                The number of expected search results.
            user_query (`str`):
                The user input based on which search will be performed.
            languages (`str`):
                The list of languages based on which search results must be filtered.
            match_text (`List`):
                A field based on which it is decided whether to perform full-text-match while performing search.
        Returns:
            results (`List[ScoredPoint]`):
                A list of `ScoredPoint` objects returned via Qdrant's search API.
    """

    filters = []

    language_mapping = {
        "Dutch": "nl",
        "English": "en",
        "French": "fr",
        "Hungarian": "hu",
        "Italian": "it",
        "Norwegian": "nb",
        "Polish": "pl",
    }

    # prepare filters to narrow down search results

    # if the `match_text` list is not empty then create filter to find exact matching text in the documents
    if match_text:
        filters.append(
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=user_query),
            )
        )

    # filter documents based on language before performing search:
    if languages:
        for lang in languages:
            filters.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(
                        value=language_mapping[lang],
                    ),
                )
            )

    # perform search and get results
    results = qdrant_client.search(
        collection_name=SEARCH_QDRANT_COLLECTION_NAME,
        query_filter=models.Filter(should=filters),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        query_vector=query_embedding,
        limit=num_results,
    )
    return results


def translate_search_result(input_sentence, user_query):
    """
    Translate a given input sentence to the required target language. The required target language is `English` by default.
    The target language can be changed to match the language that the user used to type his search query by setting the `TRANSLATE_BASED_ON_USER_QUERY` to `True`.
        Args:
            input_sentence (`str`):
                The sentence which needs to be translated into the required target language.
            user_query (`str`):
                The user input based on which the target language for translation will be determined if `TRANSLATE_BASED_ON_USER_QUERY` is set to `True`.
        Returns:
            translation (`str`):
                The final translation result obtained using Cohere's Generate API.
    """
    response = cohere_client.tokenize(text=input_sentence)

    src_detected_lang = cohere_client.detect_language(texts=[input_sentence])
    src_current_lang = src_detected_lang.results[0].language_name

    if TRANSLATE_BASED_ON_USER_QUERY:
        target_detected_lang = cohere_client.detect_language(texts=[user_query])
        target_current_lang = target_detected_lang.results[0].language_name
    else:
        target_current_lang = "English"

    if target_current_lang == src_current_lang:
        return input_sentence

    prompt = f""""
    Translate this sentence from {src_current_lang} to {target_current_lang}: '{input_sentence}'.
    
    Don't include the above prompt in the final translation. The final output should only include the translation of the input sentence.
    """

    response = cohere_client.generate(
        model=TEXT_GENERATION_MODEL,
        prompt=prompt,
        max_tokens=len(response.tokens) * 3,
        temperature=0.6,
        stop_sequences=["--"],
    )

    translation = response.generations[0].text

    return translation


def cross_lingual_document_search(
    user_input: str, num_results: int, languages, text_match
) -> List:
    """
    Wrapper function for performing search on the collection of documents for the given user query.
    Prepares query embedding, retrieves search results, checks if expected number of search results are being returned.
        Args:
            user_input (`str`):
                The user input based on which search will be performed.
            num_results (`str`):
                The number of expected search results.
            languages (`str`):
                The list of languages based on which search results must be filtered.
            text_match (`str`):
                A field based on which it is decided whether to perform full-text-match while performing search.
        Returns:
            final_results (`List[str]`):
                A list containing the final search results corresponding to the given user input.
    """
    # create an embedding for the input query
    query_embedding = embed_user_query(user_input)

    # retrieve search results
    result = search_docs_for_query(
        query_embedding,
        num_results,
        user_input,
        languages,
        text_match,
    )
    final_results = [result[i].payload["text"] for i in range(len(result))]

    # check if number of search results obtained (i.e. `final_results`) is matching with number of expected search results i.e. `num_results`
    if num_results > len(final_results):
        remaining_inputs = num_results - len(final_results)
        for input in range(remaining_inputs):
            final_results.append("")

    return final_results
