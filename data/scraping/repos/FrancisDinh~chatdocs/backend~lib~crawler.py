import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS

from threading import Thread, Lock
urls_to_visit = []
visited_urls = []
lock = Lock()

        

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

class Crawler:

    def __init__(self, url):
        self.urlOrigin = url
        
        global urls_to_visit;
        global visited_urls;
        urls_to_visit.append(url);
        self.embeddings = OpenAIEmbeddings()
        self.index_name = 'agent_vector_search'
        self.elastic_vector_search = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        textContent = soup.find('body').text

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.create_documents([textContent])
        

        for i, doc in enumerate(docs):
            doc.metadata["link"] = url
            doc.metadata["agent_id"] = '1232'
        
        db = ElasticsearchStore.from_documents(
            docs, self.embeddings, es_url="http://localhost:9200", index_name=self.index_name, 
        )

        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        if url not in visited_urls and url not in urls_to_visit:
            urls_to_visit.append(url)

    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            if (self.urlOrigin in url):
                self.add_url_to_visit(url)

    def run(self):
        while urls_to_visit:
            url = urls_to_visit.pop(0)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                visited_urls.append(url)