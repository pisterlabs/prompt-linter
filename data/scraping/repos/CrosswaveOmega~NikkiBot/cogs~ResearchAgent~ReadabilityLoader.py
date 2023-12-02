"""Web base loader class."""
import asyncio
import logging
import re
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union
import inspect
import aiohttp
import discord
import requests
import assets
from javascriptasync import require, eval_js, eval_js_a

"""This is a special loader that makes use of Mozilla's readability module. """


def is_readable(url):
    timeout = 30
    readability = require("@mozilla/readability")
    jsdom = require("jsdom")
    TurndownService = require("turndown")
    # Is there a better way to do this?
    print("attempting parse")
    out = f"""
    let result=await check_read(`{url}`,readability,jsdom);
    return result
    """
    myjs = assets.JavascriptLookup.find_javascript_file("readwebpage.js", out)
    # myjs=myjs.replace("URL",url)
    print(myjs)
    rsult = eval_js(myjs)
    return rsult


def remove_links(markdown_text):
    # Regular expression pattern to match masked links
    # pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    pattern = r"\[([^\]]+)\]\([^)]+\)"

    # Replace the masked links with their text content
    no_links_string = re.sub(pattern, r"\1", markdown_text)

    return no_links_string


async def read_article_direct(html, url):
    myfile = await assets.JavascriptLookup.get_full_pathas("readwebpage.js")
    timeout = 30

    htmls: str = str(html)

    pythonObject = {"var": htmls, "url": url}
    out = """
    let html2=await pythonObject.var
    let urlV=await pythonObject.url
    const turndownService = new TurndownService({ headingStyle: 'atx' });
    let result=await read_webpage_html_direct(html2,urlV,readability,jsdom, turndownService);
    return [result[0],result[1]];
    """

    rsult = await myfile.read_webpage_html_direct(htmls, url,timeout=45)
    output = await rsult.get_a("mark")
    header = await rsult.get_a("orig")
    serial = await header.get_dict_a()

    simplified_text = output.strip()
    simplified_text = re.sub(r"(\n){4,}", "\n\n\n", simplified_text)
    simplified_text = re.sub(r"\n\n", "\n", simplified_text)
    simplified_text = re.sub(r" {3,}", "  ", simplified_text)
    simplified_text = simplified_text.replace("\t", "")
    simplified_text = re.sub(r"\n+(\s*\n)*", "\n", simplified_text)
    return [simplified_text, serial]


async def read_article_aw(html, url):
    now = discord.utils.utcnow()
    getthread = await read_article_direct(html, url)
    result = getthread
    print(result)
    text, header = result[0], result[1]
    return text, header


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


from langchain.docstore.document import Document
import langchain.document_loaders as dl
from langchain.document_loaders import PDFMinerPDFasHTMLLoader


class ReadableLoader(dl.WebBaseLoader):
    async def scrape_all(
        self, urls: List[str], parser: Union[str, None] = None
    ) -> List[Any]:
        """Fetch all urls, then return soups for all results."""
        from bs4 import BeautifulSoup

        pdf_urls = []
        regular_urls = []

        for url in urls:
            if url.endswith(".pdf") or ".pdf?" in url:
                pdf_urls.append(url)
            else:
                regular_urls.append(url)

        results = await self.fetch_all(regular_urls)
        for pdfurl in pdf_urls:
            loader = PDFMinerPDFasHTMLLoader(pdfurl)
            data = loader.load()[0]
            results.append(data)
            regular_urls.append(pdfurl)

        final_results = []

        for i, result in enumerate(results):
            url = regular_urls[i]

            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.default_parser

                self._check_parser(parser)

            # If the URL is one of the PDF URLs, we load the PDF content
            # using PDFMinerPDFasHTMLLoader
            if url in pdf_urls:
                souped = BeautifulSoup(result.page_content, "html.parser")
            else:
                souped = BeautifulSoup(result, parser)

            try:
                clean_html = re.sub(
                    r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", result
                )
                text, header = await read_article_aw(clean_html, url)
                final_results.append((remove_links(text), souped, header))

            except Exception as e:
                text = souped.get_text(**self.bs_get_text_kwargs)
                final_results.append((text, souped, None))

        return final_results

    def _scrape(self, url: str, parser: Union[str, None] = None) -> Any:
        from bs4 import BeautifulSoup

        if parser is None:
            if url.endswith(".xml"):
                parser = "xml"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        html_doc = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()
        html_doc.encoding = html_doc.apparent_encoding
        return BeautifulSoup(html_doc.text, parser)

    def scrape(self, parser: Union[str, None] = None) -> Any:
        """Scrape data from webpage and return it in BeautifulSoup format."""

        if parser is None:
            parser = self.default_parser

        return self._scrape(self.web_path, parser)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path)
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""
        return list(self.lazy_load())

    async def aload(self) -> List[Document]:
        """Load text from the urls in web_path async into Documents."""

        results = await self.scrape_all(self.web_paths)
        docs = []
        for i,res in enumerate(results):
            text, soup, header = results[i]

            metadata = _build_metadata(soup, self.web_paths[i])
            if not "title" in metadata:
                metadata["title"] = "LoadedPDF"
            if header is not None:
                print(header["byline"])
                if "byline" in header:
                    metadata["authors"] = header["byline"]
                metadata["website"] = header.get("siteName", "siteunknown")
                metadata["title"] = header.get("title")

            metadata["sum"] = "source"
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
