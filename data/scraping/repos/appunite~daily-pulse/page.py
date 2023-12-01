import requests
from langchain.docstore.document import Document

from daily_pulse.config import config
from daily_pulse.utils.text.markdown import (
    convertBlocksToMarkdown,
    convertStandardPageInfoToMarkdown,
)


def getStandarizedPageInfo(pageObject):
    return {
        "id": pageObject["id"],
        "url": pageObject["url"],
        "name": pageObject["properties_value"]["Name"][0]["plain_text"],
        "type": pageObject["properties_value"]["Type"]["name"],
        "status": pageObject["properties_value"]["Status"]["name"],
    }


def getPageChildrenBlocks(pageId):
    url = config["GET_PAGE_CONTENT_HOOK"]
    json = {"id": pageId, "Authorization": f"Bearer {config['AUTH_CODE']}"}

    resp = requests.post(url=url, json=json)
    return resp.json()


def getPageWithDocument(pageObject):
    standarized_page = getStandarizedPageInfo(pageObject)
    page_children = getPageChildrenBlocks(standarized_page["id"])
    page_children_markdown = convertBlocksToMarkdown(page_children)
    page_header_markdown = convertStandardPageInfoToMarkdown(standarized_page)
    page_document_content = "\n".join([page_header_markdown, page_children_markdown])
    page_document = Document(page_content=page_document_content)

    return {
        "id": standarized_page["id"],
        "url": standarized_page["url"],
        "document": page_document,
    }
