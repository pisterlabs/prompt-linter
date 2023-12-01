import multiprocessing
from bs4 import BeautifulSoup
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import requests
import functools

from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS
from lib.config import Config

class MultiThreadedCrawler:

	def __init__(self, agent_id, seed_url, selector):
		self.seed_url = seed_url
		self.agent_id = agent_id
		self.selector = selector
		self.root_url = '{}://{}'.format(urlparse(self.seed_url).scheme,
										urlparse(self.seed_url).netloc)
		self.pool = ThreadPoolExecutor(max_workers=5)
		self.scraped_pages = set([])
		self.crawl_queue = Queue()
		self.crawl_queue.put(self.seed_url)
		self.embeddings = OpenAIEmbeddings()
		self.index_name = 'agent_vector_search'

	def parse_links(self, html):
		soup = BeautifulSoup(html, 'html.parser')
		Anchor_Tags = soup.find_all('a', href=True)
		for link in Anchor_Tags:
			url = link['href']
			if url.startswith('/') or url.startswith(self.root_url):
				url = urljoin(self.root_url, url)
				if url not in self.scraped_pages and '#' not in url:
					self.crawl_queue.put(url)

	def scrape_info(self, html, url):
		soup = BeautifulSoup(html, "html5lib")
		
		text = soup.select_one(self.selector).text
		text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
		docs = text_splitter.create_documents([text])
		for i, doc in enumerate(docs):
			doc.metadata["link"] = url
			doc.metadata["agent_id"] = self.agent_id
		ElasticsearchStore.from_documents(
            docs, self.embeddings, es_url=Config.ES_URL, index_name=self.index_name, 
        )

		return

	def post_scrape_callback(self, res):
		result = res.result()
	
		if result and result.status_code == 200:
			self.parse_links(result.text)
			self.scrape_info(result.text, result.url)

	def scrape_page(self, url):
		try:
			res = requests.get(url, timeout=(3, 30))
			return res
		except requests.RequestException:
			return

	def run_web_crawler(self):
		while True:
			try:
				print("\n Name of the current executing process: ",
					multiprocessing.current_process().name, '\n')
				target_url = self.crawl_queue.get(timeout=60)
				if target_url not in self.scraped_pages:
					print("Scraping URL: {}".format(target_url))
					self.current_scraping_url = "{}".format(target_url)
					self.scraped_pages.add(target_url)
					job = self.pool.submit(self.scrape_page, target_url)
					job.add_done_callback(self.post_scrape_callback)


			except Empty:
				return
			except Exception as e:
				print(e)
				continue

	def info(self):
		print('\n Seed URL is: ', self.seed_url, '\n')
		print('Scraped pages are: ', self.scraped_pages, '\n')

