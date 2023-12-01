import os
import pathlib
from datetime import datetime
from typing import Optional, Type

import requests
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.vectorstores import Chroma

import bs4
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

os.environ['OPENAI_API_KEY'] = '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'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


from typing import Optional

class WebsiteSearchTool(BaseTool):
    name = "website_search"
    description = "Useful when you need to get search results from the website."
    llm = ChatOpenAI(temperature=0)

    def scrape_website(self, url, css_class, timeout=10):
        """
        Scrapes a website using Selenium and waits for a specific class to be rendered.

        Parameters:
        - url (str): The URL of the website to scrape.
        - css_class (str): The CSS class to wait for before scraping.
        - timeout (int, optional): Maximum time to wait for the class to render. Default is 10 seconds.

        Returns:
        - str: The source HTML of the website after waiting for the class to render.
        """

        # Create a new instance of the Firefox driver (you can use Chrome or other drivers if you prefer)
        driver = webdriver.Firefox()

        # Navigate to the desired URL
        driver.get(url)

        # Wait for the specified class to be present on the page
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, css_class))
            WebDriverWait(driver, timeout).until(element_present)
        except:
            print("Timed out waiting for the element to render.")

        # Get the source HTML of the page
        html = driver.page_source

        # Quit the driver
        driver.quit()

        return html

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # scrape the website https://www.anuga.com/search/?q={query}

        # Transform query into URL Parameter
        query = query.replace(" ", "+")

        # get the html
        html = self.scrape_website(f'https://www.anuga.com/search/?q={query}', 'search-results')

        # Extract class 'search-results' from html
        soup = bs4.BeautifulSoup(html, 'html.parser')
        search_results = soup.find_all(class_='search-results')
        search_responses = []

        # Loop for every div directly inside search_results, return 3 results
        for i, result in enumerate(search_results[0].find_all('div', recursive=False)[:3]):
            # Get the title from .search-results__item-title > a
            title = result.find(class_='search-results__item-title').find('a').text

            # Get the description from all p tags from .search-results__item-text > p except p.search-results__item-read-more
            description = result.find(class_='search-results__item-text').find_all('p', class_='search-results__item-text')[0].text

            # Get the URL and Link title from p.search-results__item-read-more > a
            url = result.find(class_='search-results__item-text').find_all('p', class_='search-results__item-text')[1].find('a')['href']
            link_title = result.find(class_='search-results__item-text').find_all('p', class_='search-results__item-text')[1].find('a').text

            # Add the result to the search_response dict
            search_responses[i] = {
                'title': title,
                'description': description,
                'url': url,
                'link_title': link_title
            }

        return str(search_responses)


    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("star_coder does not support async")