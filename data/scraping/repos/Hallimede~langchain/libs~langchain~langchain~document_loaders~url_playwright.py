"""Loader that uses Playwright to load a page, then uses unstructured to load the html.
"""
import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PlaywrightURLLoader(BaseLoader):
    """Loader that uses Playwright and to load a page and unstructured to load the html.
    This is useful for loading pages that require javascript to render.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        headless (bool): If True, the browser will run in headless mode.
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        headless: bool = True,
        remove_selectors: Optional[List[str]] = None,
    ):
        """Load a list of URLs using Playwright and unstructured."""
        try:
            import playwright  # noqa:F401
        except ImportError:
            raise ImportError(
                "playwright package not found, please install it with "
                "`pip install playwright`"
            )

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headless = headless
        self.remove_selectors = remove_selectors

    def load(self) -> List[Document]:
        """Load the specified URLs using Playwright and create Document instances.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """
        from playwright.sync_api import sync_playwright
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            for url in self.urls:
                try:
                    page = browser.new_page()
                    page.goto(url)

                    for selector in self.remove_selectors or []:
                        elements = page.locator(selector).all()
                        for element in elements:
                            if element.is_visible():
                                element.evaluate("element => element.remove()")

                    page_source = page.content()
                    elements = partition_html(text=page_source)
                    text = "\n\n".join([str(el) for el in elements])
                    metadata = {"source": url}
                    docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(
                            f"Error fetching or processing {url}, exception: {e}"
                        )
                    else:
                        raise e
            browser.close()
        return docs

    async def aload(self) -> List[Document]:
        """Load the specified URLs with Playwright and create Documents asynchronously.
        Use this function when in a jupyter notebook environment.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """
        from playwright.async_api import async_playwright
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            for url in self.urls:
                try:
                    page = await browser.new_page()
                    await page.goto(url)

                    for selector in self.remove_selectors or []:
                        elements = await page.locator(selector).all()
                        for element in elements:
                            if await element.is_visible():
                                await element.evaluate("element => element.remove()")

                    page_source = await page.content()
                    elements = partition_html(text=page_source)
                    text = "\n\n".join([str(el) for el in elements])
                    metadata = {"source": url}
                    docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(
                            f"Error fetching or processing {url}, exception: {e}"
                        )
                    else:
                        raise e
            await browser.close()
        return docs

    async def aload_with_subpages(self, n) -> List[Document]:
        """Load the specified URLs with Playwright and create Documents asynchronously.
        Use this function when in a jupyter notebook environment.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """
        from playwright.async_api import async_playwright

        docs: List[Document] = list()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            docs = await self.crawl_pages(browser, self.urls, True)
            await browser.close()
        return docs

    async def crawl_pages(self, browser, urls, is_root, n) -> List[Document]:
        from unstructured.partition.html import partition_html
        docs: List[Document] = list()
        for url in urls:
            print(f"Loading {url}")
            try:
                page = await browser.new_page()
                await page.goto(url)

                if is_root:

                    # Extract hyperlinks from the page
                    links = await page.evaluate('''() => {
                        return Array.from(document.querySelectorAll('a'))
                            .map(a => a.href);
                    }''')

                    # Select 10 hyperlinks under the same domain
                    url_start = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

                    links_to_be_crawled = set()

                    for link in links:
                        if link.startswith(url_start) and len(links_to_be_crawled) < n:
                            links_to_be_crawled.add(link)
                    
                    print("Sub links to be crawled:")
                    print(links_to_be_crawled)
                    docs.extend(await self.crawl_pages(browser, list(links_to_be_crawled), False))

                for selector in self.remove_selectors or []:
                    elements = await page.locator(selector).all()
                    for element in elements:
                        if await element.is_visible():
                            await element.evaluate("element => element.remove()")

                page_source = await page.content()
                elements = partition_html(text=page_source)
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": url}
                docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(
                        f"Error fetching or processing {url}, exception: {e}"
                    )
                else:
                    raise e
        return docs
