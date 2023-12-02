from langchain.document_loaders.xorbits import XorbitsLoader
from typing import Any, Dict

from IPython.display import HTML
import pandas as pd


def gsutil_uri_to_gcs_url(gsutil_uri: str) -> str:
    """Converts a gsutil URI to a GCS URL.

    Args:
        gsutil_uri (str): The gsutil URI to be converted, e.g., "gs://bucket-name/object-name".

    Returns:
        str: The converted GCS URL, e.g., "https://storage.googleapis.com/bucket-name/object-name".
    """
    return gsutil_uri.replace("gs://", "https://storage.googleapis.com/")


def format_results(results: Dict, print_source=False, return_html=True) -> Any:
    """
    Formats and prints the results of a question-answering query.
    Args:
        results: The results of a question-answering query.
    Returns:
        None
    """
    sep = "*" * 79
    docs = results['source_documents']
    # Display settings for columns
    pd.set_option('display.max_colwidth', 80)
    # Prepare the header for the table
    print(sep)
    print(f"Answer: {results['result']}")
    print(f"Used {len(docs)} relevant documents.")
    print(sep)

    # Populate DataFrame
    records = [
        (gsutil_uri_to_gcs_url(doc.metadata['source']), 
         doc.page_content[:100] + "...") 
        for doc in docs
    ]
    df = pd.DataFrame.from_records(records, columns=["Source", "Preview"])

    if print_source:
        for idx, doc in enumerate(docs):
            print(sep)
            print(f"Document: {idx}")
            print(f"Source: {gsutil_uri_to_gcs_url(doc.metadata['source'])}")
            print("Content:")
            print(doc.page_content)
    
    if return_html:
        return HTML(df.to_html(render_links=True, escape=False))

    return df
