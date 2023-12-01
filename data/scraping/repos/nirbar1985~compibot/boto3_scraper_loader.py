# pylint: disable=print-used, missing-timeout

from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


class Boto3ScraperLoader():

    BASE_URL_FORMAT = 'https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/{service_name}.html'

    def __init__(self, service_name):
        self.__service_name = service_name
        self.__base_url_service = self.BASE_URL_FORMAT.format(service_name=self.__service_name)

    # def __init__(self):
    #     self.scraper = Boto3Scraper()
    @staticmethod
    def get_soup(url):
        response = requests.get(url, headers=HEADERS)
        return BeautifulSoup(response.content, 'html.parser')

    def load(self) -> List[Document]:
        soup = Boto3ScraperLoader.get_soup(self.__base_url_service)

        # Find all <a> tags with href starting with the service name
        links = soup.find_all('a', href=True)
        service_api_links = [link['href'] for link in links if link['href'].startswith(self.__service_name)]
        print('service api links are:\n ' + str(service_api_links))

        url_without_suffix = self.__base_url_service.replace('.html', '')

        full_urls = [urljoin(url_without_suffix, link) for link in service_api_links]
        full_urls.append(self.__base_url_service)

        loader = WebBaseLoader(full_urls)
        loader.requests_per_second = 50
        return loader.load()
