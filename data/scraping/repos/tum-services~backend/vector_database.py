from bs4 import BeautifulSoup
import re
import copy
import os
from langchain.vectorstores import Pinecone
from langchain.vectorstores.pinecone import Document
from langchain.embeddings import OpenAIEmbeddings
import markdownify
from langchain.document_loaders import WebBaseLoader
from os import listdir
from os.path import isfile, join

PATH = "sites"

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")


def fix_whitespaces(text):
    text = re.sub('\r{2,}',' ', text)
    text = re.sub('\t{2,}',' ', text)
    text = re.sub('\n{2,}', ' ', text)
    return re.sub(' {2,}', ' ', text)


def save_chunks(documents):
    vectorstore = Pinecone.from_documents(
        documents=documents, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
    )
    retriever = vectorstore.as_retriever()
    print(f"Content saved to pinecone index {PINECONE_INDEX_NAME}")
    return retriever


def get_description(whole_soup, div):
    headings = []
    if whole_soup.find("nav", class_='breadcrumbs').find_all('li') is not None:
        headings += [breadcrumbs.text for breadcrumbs in whole_soup.find("nav", class_='breadcrumbs').find_all('li')]
    if whole_soup.find('h1') is not None:
        headings.append(whole_soup.find('h1').text)
    headings += [div.find_previous(h).text for h in ['h2', 'h3', 'h4', 'h5', 'h6'] if
                 div.find_previous(h) is not None]
    description = ">".join(headings)
    return fix_whitespaces(description)


def get_documents_content(soup, url, title, ignore):
    content = soup.find("div", class_="content")
    top_layer_divs = [child for child in content.children if
                      child.name == 'div' and not (ignore & set(child.get('class', [])))]
    return get_documents(content, soup, url, title, top_layer_divs)


def get_documents_sidebar(soup, url, title, ignore):
    sidebar = soup.find("div", class_="sidebar")
    if sidebar is None:
        return []
    aside_elements = [child for child in sidebar.children if
                      child.name == 'aside' and not (ignore & set(child.get('class', [])))]
    top_layer_divs = []
    for aside in aside_elements:
        top_layer_divs += aside.find_all('div')
    return get_documents(sidebar, soup, url, title, top_layer_divs, False)


def get_chunk(website, url):
    soup = BeautifulSoup(website, 'html.parser')
    documents = []
    if soup.find("title") is None:
        title = "No title"
    else:
        title = soup.find("title").text.strip()
    ignore = {"frame-type-carousel"}
    documents += get_documents_content(soup, url, title, ignore)
    documents += get_documents_sidebar(soup, url, title, ignore)
    return documents


def get_documents(current_soup, whole_soup, url, title, top_layer_divs, heading=True, overlapping=1):
    documents = []
    if current_soup is None:
        return documents

    for i in range(len(top_layer_divs)):
        description = get_description(whole_soup, top_layer_divs[i])
        text = ""
        if heading:
            text = "\nDescription: " + description + "\n"
        for j in range(i, min(i + overlapping, len(top_layer_divs))):
            tables = top_layer_divs[j].find_all('table')
            for table in tables:
                try:
                    markdown_table = markdownify.markdownify(str(table))
                    text += markdown_table.replace('\\*', '')
                    table.decompose()
                except TypeError:
                    pass
            text += "\nText: " + top_layer_divs[j].text + "\n"
        text = fix_whitespaces(text)
        document = Document(
            page_content=text,
            metadata={"source": url, "description": description, "title": title, "text":  text}
        )
        documents.append(document)
    return documents


def absence_chunks():
    url = "https://www.tum.de/studium/im-studium/das-studium-organisieren/beurlaubung"
    description = "Beurlaubung\nLeave of absence\nAntrag auf Urlaub"
    text = description + '''
        In diesem Chat kannst du eine Beurlaubung beantragen.'''
    title = "Beurlaubung"
    doc1 = Document(
        page_content=text,
        metadata={"source": url, "description": description, "title": title, "text": text, "wizzard": 0}
    )
    return [doc1]

#if __name__ == '__main__':
#    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]
#    for file in files:
#        with open(f"{PATH}/{file}", "r") as f:
#            website = f.read()
#            save_website(website, file)
    





