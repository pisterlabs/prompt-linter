from langchain.document_loaders import SeleniumURLLoader
from store_docs.data_sources import url_sources, md_sources
import requests
from langchain.document_loaders import UnstructuredMarkdownLoader


def fetch_url_sources():
    urls = [obj.url for obj in url_sources]
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    return data


def fetch_md_sources():
    import os

    filename = "temp/output.md"
    all_data = []
    for obj in md_sources:

        # Send a GET request to the URL and get the content
        response = requests.get(obj.url)
        content = response.content

        # Save the content to a file
        with open(filename, "wb") as file:
            file.write(content)

        loader = UnstructuredMarkdownLoader(filename)
        all_data += loader.load()

    os.remove(filename)

    return all_data
