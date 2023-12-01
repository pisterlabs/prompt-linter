from seaplane import task
from langchain.text_splitter import RecursiveCharacterTextSplitter
from seaplane.model import Vector
from seaplane.vector import vector_store
import uuid
import requests
from bs4 import BeautifulSoup
import os
from langchain.embeddings import OpenAIEmbeddings
import logging


@task(id="documentation-processors", type="compute")
def process_docs(data):
    # initiate Langhcain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    # recreate the index, this whipes the existing index and starts over Update
    # <YOUR-INDEX> with your index name. We recommend companyname-docusaurus.
    # This example uses OpenAI embeddings which have a dimention of 1536. Update
    # the dimensions if you use other embeddings. For example, the Seaplane
    # embeddigns have a dimension of 768
    index_name = "<YOUR-INDEX>"
    vector_store.recreate_index(index_name, 1536)

    # Get the URL of the sitemap.xml from the API request
    url = data["url"]

    # Fetch the XML data from the URL
    response = requests.get(url)
    xml_data = response.text

    # Parse the XML data using BeautifulSoup
    soup = BeautifulSoup(xml_data, "xml")

    # create openAI embeddign
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Find all the 'loc' elements
    pages = soup.find_all("loc")

    # loop through pages
    for page in pages:
        url = page.get_text()
        # check if this is part of the docs pages or something else. Make sure
        # to extend this if you have mulitple content doc plugins running. For
        # example you can check for /guides or /tutorials as well.
        if "docs" in url and "tags" not in url:
            # fetch the current page
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # find all divs on page
            selected_divs = soup.find_all(
                "div", {"class": "theme-doc-markdown markdown"}
            )

            # skip pages without main content div
            try:
                # select the main content div
                content = selected_divs[0].text
            except IndexError:
                logging.error(f"skipping: {url}")

            # create text splitter
            texts = text_splitter.create_documents([content])

            # add URL as metadata to enable source link in chatbot
            for idx, text in enumerate(texts):
                texts[idx].metadata["url"] = str(url)

            # crete embeddings
            vectors = embedding.embed_documents([page.page_content for page in texts])

            # construct vectors
            vectors = [
                Vector(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    metadata={
                        "page_content": texts[idx].page_content,
                        "metadata": texts[idx].metadata,
                    },
                )
                for idx, vector in enumerate(vectors)
            ]

            # insert into vector store.
            vector_store.insert(index_name, vectors)

    return "done"
