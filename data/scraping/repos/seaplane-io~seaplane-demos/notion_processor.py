from seaplane import task
from langchain.text_splitter import RecursiveCharacterTextSplitter
from seaplane.model import Vector
from seaplane.vector import vector_store
import uuid
from seaplane.integrations.langchain import seaplane_embeddings
import os

# import helper files to scrape Notion API
from helper_files import get_all_pages, get_page, get_page_content


@task(id="notion-processor", type="compute")
def process_notion(data):
    # create vector store if it does not yet exists 768 dimenions for seaplane embeddings
    vector_store.create_index("notion-search", 768)

    # initiate Langhcain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    # set the headers
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_KEY')}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    # get all pages we have access to with the integration
    pages = get_all_pages(headers)

    # loop through all available pages
    for page in pages:
        for i in page["results"]:
            # get page from Notion
            page = get_page(i["id"], headers)

            # extract page content
            page_content = get_page_content(page)

            # check for empty page
            if page_content == "" or page_content == " ":
                # move on nothing to do here
                continue

            # create documents from page content
            texts = text_splitter.create_documents([page_content])

            # embed documents
            vectors = seaplane_embeddings.embed_documents(
                [page.page_content for page in texts]
            )

            # create vectors with data component
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

            # insert into vector store
            vector_store.insert("notion-search", vectors)

    # show that we are done
    return "done"
