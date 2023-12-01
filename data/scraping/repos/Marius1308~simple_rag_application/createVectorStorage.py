import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from pageLinks import root, pages
import requests
from langchain.document_loaders import BSHTMLLoader
from html.parser import HTMLParser


def cutNewLines(page_content):
    while "\n\n" in page_content:
        page_content = page_content.replace("\n\n", "\n")
    while page_content[-1] == "\n":
        page_content = page_content[:-1]
    while page_content[0] == "\n":
        page_content = page_content[1:]
    return page_content


def addNewLine(page_content, chunk_size):
    newPageContent = ""
    buffer = chunk_size / 10
    added = False
    for index, char in enumerate(page_content):
        suffix = ""
        if (
            index % chunk_size < buffer or index % chunk_size > chunk_size - buffer
        ) and index != 0:
            if not added and char in [".", "!", "?"]:
                suffix = "\n\n"
                added = True
        if not added and index % chunk_size == buffer:
            suffix = "\n\n"
            added = True
        if index % chunk_size == buffer + 1:
            added = False
        newPageContent += char + suffix
    return newPageContent


parsedPages = []


class MyHTMLParser(HTMLParser):
    parsedPages = []

    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
            # Check the list of defined attributes.
            for name, value in attrs:
                # If href is defined, print it.
                if name == "href":
                    if value.startswith("/docs/") and not value.endswith(".html"):
                        if value not in self.parsedPages:
                            newPage = value
                            if value.endswith("/"):
                                newPage = value[:-1]
                            self.parsedPages.append(newPage.replace("/docs/", ""))

    def resetPages(self):
        self.parsedPages = []


def addPages(pages, root):
    parser = MyHTMLParser()
    paths = []
    additionalPages = []
    for page in pages:
        response = requests.get(f"{root}/{page}")
        if response.status_code != 200:
            continue

        pageName = page.replace("/", "-")
        pagePath = f"./langchainPages/html/{pageName}.html"

        responseText = response.content.decode("utf-8")
        text_file = open(pagePath, "w")
        text_file.write(responseText)
        text_file.close()
        paths.append(pagePath)
        parser.resetPages()
        parser.feed(responseText)
        additionalPages += parser.parsedPages
    return paths, additionalPages


root = "https://python.langchain.com/docs"
pages = [
    "get_started/introduction",
    "get_started/installation",
    "get_started/quickstart",
]
addedPages = 0
paths = []

hasNewPages = True
while hasNewPages:
    hasNewPages = False
    newPaths, newPages = addPages(pages[addedPages:], root)
    addedPages = len(pages)
    print(addedPages)
    paths += newPaths
    for page in newPages:
        if page not in pages:
            pages.append(page)
            hasNewPages = True


for path in paths:
    loader = BSHTMLLoader(path)
    raw_documents = loader.load()
    for index, document in enumerate(raw_documents):
        raw_documents[index].page_content = addNewLine(
            cutNewLines(document.page_content), 1000
        )
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db2 = Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        persist_directory="./langchainPages/db/chroma_db",
    )
