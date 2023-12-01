import asyncio
from typing import Tuple, List
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import requests
from newspaper import Article
from langchain.docstore.document import Document
import aiohttp
from streamlit.delta_generator import DeltaGenerator
from DataAccess import DataAccess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="crawler.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


class Crawler:
    def __init__(self, data_access: DataAccess):
        self.ui_update_in_progress = False
        self.data_access = data_access
        self.urls = None
        self.counter = 0
        self.counter_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(
            20
        )  # limit to 10 concurrent tasks, adjust as needed
        self.sitemap_gen_semaphore = asyncio.Semaphore(20)

    def get_url_count(self):
        return len(self.urls)

    def load_urls_from_file(self):
        try:
            with open("urls.txt", "r") as file:
                return [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            return None

    def get_sitemap_urls(self, sitemap_url: str, onlyEnglish: bool):
        # Need to fix issue where it doesn't detect if onlyEnglish has been changed and will use the wrong cache file.
        urls = self.load_urls_from_file()
        if urls is None:
            r = requests.get(sitemap_url)
            soup = BeautifulSoup(r.text, "xml")
            if onlyEnglish:
                urls = [
                    loc.text for loc in soup.find_all("loc") if "locale" not in loc.text
                ]
            else:
                urls = [loc.text for loc in soup.find_all("loc")]
            with open("urls.txt", "w") as file:
                for url in urls:
                    file.write("%s\n" % url)
        return urls

    async def extract_page_content(self, url: str, session: ClientSession):
        logging.info(f"Extracting content from URL: {url}")
        try:
            async with session.get(url) as response:
                html_content = await response.text()

            # Use newspaper's Article for parsing
            article = Article(url)
            article.set_html(html_content)
            article.parse()
            content = article.text
            title = article.title
            logging.info(f"Successfully extracted content from URL: {url}")
            return url, content, title
        except Exception as e:
            print(f"Timeout error for URL: {url}")
            logging.error(f"Timeout or some other error extracting URL {url}: {e}")
            return url, "", ""

    async def async_chunk_page(
        self, url: str, session: ClientSession
    ) -> Tuple[str, List[str], str]:
        url, content, title = await self.extract_page_content(url, session)
        chunks = self.data_access.splitter.split_text(content)
        return url, chunks, title

    async def handle_url(
        self, url: str, progress_bar: DeltaGenerator, session: ClientSession
    ):
        async with self.semaphore:  # this will wait if there are already too many tasks running:
            url, chunks, title = await self.async_chunk_page(url, session)
            if len(chunks) == 1 and len(chunks[0]) < 500:
                print(f"Skipping page {url} with only 1 chunk under 500 characters.")
                logging.info(
                    f"Skipping page {url} with only 1 chunk under 500 characters."
                )
            else:
                page_docs = [
                    Document(
                        page_content=chunk,
                        metadata={"url": url, "title": title},
                    )
                    for chunk in chunks
                ]
                # async transform_documents is not available yet
                split_docs = self.data_access.splitter.transform_documents(page_docs)
                vectorStore = self.data_access.getVectorStore()
                # vectorStore.aadd_documents(split_docs) isn't yet implemented. We will work around it.
                # await vectorStore.aadd_documents(split_docs)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, vectorStore.add_documents, split_docs)
                logging.info(f"Written to database, URL: {url}, Title: {title}")

        async with self.counter_lock:
            self.counter += 1
            # Only call the UI update if no update is currently in progress to avoid blocking threads with UI updates since it's okay if some get skipped:
            if not self.ui_update_in_progress:
                self.ui_update_in_progress = True
            await self.update_UI(self.counter, progress_bar)

    async def update_UI(self, counter: int, progress_bar: DeltaGenerator):
        total_url_count = self.get_url_count()
        percentage_completion = ((counter + 1) / total_url_count) * 100
        print(f"Completed {counter} out of {total_url_count} in total")
        print(f"Processing... Completion: {percentage_completion:.2f}%")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        progress_bar.progress(
            int(percentage_completion),
            text=f"Processing... Completion: {percentage_completion:.2f}%",
        )
        self.ui_update_in_progress = False

    async def process_urls(self, progress_bar: DeltaGenerator):
        timeout = aiohttp.ClientTimeout(
            total=30
        )  # 10 seconds timeout for the entire request process
        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:  # If needed, use session for HTTP requests
            tasks = [self.handle_url(url, progress_bar, session) for url in self.urls]
            await asyncio.gather(*tasks)

    def async_crawl_and_ingest(self, sitemap_url: str, progress_bar: DeltaGenerator):
        if self.urls is None:
            # TODO: Need to cache the following step:
            self.urls = self.get_sitemap_urls(sitemap_url, onlyEnglish=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_urls(progress_bar))
        finally:
            loop.close()

    def async_crawl_and_ingest_list(
        self, sitemap_url_list: list[str], progress_bar: DeltaGenerator
    ):
        if self.urls is None:
            # TODO: Need to cache the following step:
            all_urls = [
                url
                for sitemap in sitemap_url_list
                for url in self.get_sitemap_urls(sitemap, onlyEnglish=True)
            ]
            length_of_urls = len(all_urls)
            unique_urls = list(set(all_urls))
            logging.info(f"Length of all_urls is: {length_of_urls}")
            print(f"Length of all_urls is: {length_of_urls}")
            length_of_unique_urls = len(unique_urls)
            print(f"Length of unique all_urls is: {length_of_unique_urls}")
            logging.info(f"Length of unique all_urls is: {length_of_unique_urls}")
            self.urls = unique_urls

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_urls(progress_bar))
        finally:
            loop.close()
