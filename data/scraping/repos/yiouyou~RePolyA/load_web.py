from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import NewsURLLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
# import nest_asyncio
# nest_asyncio.apply()


##### WebBaseLoader
def load_urls_to_docs(web_paths: list[str]):
    loader = WebBaseLoader(
        web_paths=web_paths,
        verify_ssl=False,
        continue_on_failure=True,
        requests_per_second = 2,
        raise_for_status=False,
    )
    docs = loader.aload()
    return docs


##### NewsURLLoader
def load_news_url_to_docs(urls: list[str]):
    loader = NewsURLLoader(
        urls=urls,
        text_mode=True,
        nlp=True,
        show_progress_bar=True,
    )
    docs = loader.aload()
    return docs


##### AsyncChromiumLoader
def load_async_chromium_to_docs(urls: list[str]):
    loader = AsyncChromiumLoader(
        urls=urls,
    )
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed


##### RecursiveUrlLoader
def load_recursive_url_to_docs(url: str, exclude_dirs: list[str] = []):
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=2,
        use_async=True,
        exclude_dirs=exclude_dirs,
        extractor=lambda x: BeautifulSoup(x, "html.parser").text,
    )
    docs = loader.load()
    return docs


##### SitemapLoader
def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()
    return str(content.get_text())

def load_sitemap_to_docs(web_path: str, filter_urls: list[str] = []):
    loader = SitemapLoader(
        web_path=web_path,
        filter_urls=filter_urls,
        parsing_function=remove_nav_and_header_elements,
    )
    # Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
    loader.requests_kwargs = {"verify": False}
    loader.requests_per_second = 2
    docs = loader.aload()
    return docs

