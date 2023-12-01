"""Helper functions for document embedding."""
import os
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import CSVLoader
from bs4 import BeautifulSoup

# pylint: disable=line-too-long,invalid-name

def add_documents(document_loader, chroma_instance):
    """Adds documents from a langchain loader to the database"""
    documents = document_loader.load()
    # The customized splitters serve to be able to break at sentence level if required.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=225, separators= ["\n\n", "\n", ".", ";", ",", " ", ""])
    texts = text_splitter.split_documents(documents)
    chroma_instance.add_documents(texts)

def sanitize_blog_post(content: BeautifulSoup) -> str:
    """Find all unneeded elements in the BeautifulSoup object."""
    widget_elements = content.find_all('div', {"class": "widget-area"})
    nav_elements = content.find_all('nav')
    top_elements = content.find_all('div', {"class": "top-bar"})
    author = content.find_all('div', {"class": "saboxplugin-wrap"})
    related = content.find_all('div', {"class": "rp4wp-related-posts"})

    # Remove them from the BeautifulSoup object
    for element in nav_elements+widget_elements+top_elements+author+related :
        element.decompose()

    return re.sub("\n+","\n", str(content.get_text()))

def sanitize_documentx_page(content: BeautifulSoup) -> str:
    """Find all unneeded elements in the BeautifulSoup object."""
    # remove some areas
    syntax_element = content.find('div', {"id": "i-syntax-section-content"})
    requirements_element = content.find('div', {"id": "i-requirements-section-content"})
    see_also_element = content.find('div', {"id": "i-seealso-section-content"})

    # Remove them from the BeautifulSoup object
    elements = [element for element in [syntax_element, requirements_element, see_also_element] if element is not None]

    for element in elements:
        element.decompose()

    # Now find content div element
    div_element = content.find('div', {"class": "i-body-content"})
    return re.sub("\n+","\n", str(div_element.get_text()))

def sanitize_content_page(content: BeautifulSoup) -> str:
    """Find all unneeded elements in the BeautifulSoup object."""
    # Find content div element
    div_element = content.find('div', {"id": "main-content"})
    return re.sub("\n+","\n", str(div_element.get_text()))


# Create embeddings instance
embeddings = OpenAIEmbeddings()

# Create Chroma instance
instanceEN = Chroma(embedding_function=embeddings,
                  persist_directory=".\\combit_en")

instanceDE = Chroma(embedding_function=embeddings,
                  persist_directory=".\\combit_de")


def add_sitemap_documents(web_path, filter_urls, parsing_function, chroma_instance):
    """Adds all pages given in the web_path. Allows to filter and parse/sanitize the pages."""
    if os.path.isfile(web_path):
        # If it's a local file path, use the SitemapLoader with is_local=True
        sitemap_loader = SitemapLoader(web_path=web_path, filter_urls=filter_urls, parsing_function=parsing_function, is_local=True)
    else:
        # If it's a web URL, use the SitemapLoader with web_path
        sitemap_loader = SitemapLoader(web_path=web_path, filter_urls=filter_urls, parsing_function=parsing_function)

    sitemap_loader.session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"
    add_documents(sitemap_loader, chroma_instance)

# add EN .NET help from docu.combit.net
add_sitemap_documents('.\\input_en\\sitemap_net.xml',
                      [],
                      sanitize_documentx_page,
                      instanceEN)

# add EN sitemap
add_sitemap_documents('https://www.combit.com/page-sitemap.xml',
                      [],
                      sanitize_content_page,
                      instanceEN)

# add EN designer help from docu.combit.net
add_sitemap_documents('.\\input_en\\sitemap_designer.xml',
                      [],
                      None,
                      instanceEN)

# add EN programmer's reference from docu.combit.net
add_sitemap_documents('.\\input_en\\sitemap_progref.xml',
                      [],
                      None,
                      instanceEN)

# add EN Report Server reference from docu.combit.net
add_sitemap_documents('.\\input_en\\sitemap_reportserver.xml',
                      [],
                      None,
                      instanceEN)

# add EN AdHoc Designer reference from docu.combit.net
add_sitemap_documents('.\\input_en\\sitemap_adhoc.xml',
                      [],
                      None,
                      instanceEN)

# add EN Blog
add_sitemap_documents('https://www.combit.blog/post-sitemap.xml',
                      ['https://www.combit.blog/en/'],
                      sanitize_blog_post,
                      instanceEN)

# add KB dump
csv_loader = CSVLoader('.\\input_en\\kb_sanitized.csv',
                   source_column='link',
                   encoding='utf-8',
                   csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': ['title','raw','link']
                    })
add_documents(csv_loader, instanceEN)

# add DE .NET help from docu.combit.net
add_sitemap_documents('.\\input_de\\sitemap_net.xml',
                      [],
                      sanitize_documentx_page,
                      instanceDE)

# add DE sitemap
add_sitemap_documents('https://www.combit.de/page-sitemap.xml',
                      ['https://www.combit.net/reporting-tool/'],
                      sanitize_content_page,
                      instanceDE)

# add DE designer help from docu.combit.net
add_sitemap_documents('.\\input_de\\sitemap_designer.xml',
                      [],
                      None,
                      instanceDE)

# add DE programmer's reference from docu.combit.net
add_sitemap_documents('.\\input_de\\sitemap_progref.xml',
                      [],
                      None,
                      instanceDE)

# add DE Report Server reference from docu.combit.net
add_sitemap_documents('.\\input_de\\sitemap_reportserver.xml',
                      [],
                      None,
                      instanceDE)

# add DE AdHoc Designer reference from docu.combit.net
add_sitemap_documents('.\\input_de\\sitemap_adhoc.xml',
                      [],
                      None,
                      instanceDE)

# add DE Blog
add_sitemap_documents('https://www.combit.blog/post-sitemap.xml',
                      ['https://www.combit.blog/de/'],
                      sanitize_blog_post,
                      instanceDE)

# add KB dump
csv_loader = CSVLoader('.\\input_de\\kb_sanitized.csv',
                   source_column='link',
                   encoding='utf-8',
                   csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': ['title','raw','link']
                    })
add_documents(csv_loader, instanceDE)

instanceEN.persist()
instanceEN = None
instanceDE.persist()
instanceEN = None
