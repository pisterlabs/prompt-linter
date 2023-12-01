import streamlit as st
import weaviate
from streamlit_examples.utils.cohere import cohere_api_key


@st.cache_resource(
    show_spinner="Connecting to Weaviate...",
    validate=lambda client: client.is_ready(),
)
def connect_weaviate():
    # Connect to the Weaviate demo database containing 10M wikipedia vectors
    # This uses a public READ-ONLY Weaviate API key
    auth_config = weaviate.auth.AuthApiKey(
        api_key="76320a90-53d8-42bc-b41d-678647c6672e"
    )
    client = weaviate.Client(
        url="https://cohere-demo.weaviate.network/",
        auth_client_secret=auth_config,
        additional_headers={
            "X-Cohere-Api-Key": cohere_api_key,
        },
    )

    return client


def search_wikipedia(query, results_lang="en", limit=5):
    """
    Query the vectors database and return the top results.


    Parameters
    ----------
        query: str
            The search query

        results_lang: str (optional)
            Retrieve results only in the specified language.
            The demo dataset has those languages:
            en, de, fr, es, it, ja, ar, zh, ko, hi

    """

    client = connect_weaviate()

    nearText = {"concepts": [query]}
    properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]

    # To filter by language
    if results_lang != "":
        where_filter = {
            "path": ["lang"],
            "operator": "Equal",
            "valueString": results_lang,
        }
        response = (
            client.query.get("Articles", properties)
            .with_where(where_filter)
            .with_near_text(nearText)
            .with_limit(limit)
            .do()
        )

    # Search all languages
    else:
        response = (
            client.query.get("Articles", properties)
            .with_near_text(nearText)
            .with_limit(limit)
            .do()
        )

    result = response["data"]["Get"]["Articles"]

    return result
