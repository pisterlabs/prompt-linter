from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from tqdm import tqdm
from typing import List, Dict
from urllib.parse import urljoin
import requests

from dags.modules.database.pymongo import PymongoClient


class KhutodayCollector:
    today_date = date.today()
    tomarrow_date = date.today() + timedelta(days=1)

    def __init__(self):
        self.links=[]
        self.documents = []
    
    def collect(
        self,
        start_date: datetime.date = today_date,
        end_date: datetime.date = tomarrow_date,
    ):
        self.links = self._get_links(start_date, end_date)
        self.documents = self._get_documents(self.links)
        return self.documents
    
    def upload_db(self, db_host="localhost", db_port=27017):
        client = PymongoClient(host=db_host, port=db_port)
        client.insert_documents(self.documents)

    def _get_documents(self, links):
        documents = []
        for link in tqdm(links):
            response = requests.get(link)
            if "STACKS" not in response.json().keys():
                print(f"{link} is not valid.")
                continue
            stacks = response.json()["STACKS"]
            for stack in stacks:
                #If page content is empty.
                if not stack[2]:
                    continue
                item =                 {
                    "page_content": stack[2],
                    "page_url": stack[3],
                    "collected_at": stack[1],
                }
                documents.append(item)
        return documents
    
    def _get_links(self, start_date, end_date, host = "http://163.180.142.196:9090/today_api/"):
        links = []
        delta = end_date - start_date
        for i in range(delta.days):
            day = start_date+ timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            link = urljoin(base = host, url=day_str)
            links.append(link)
        return links



    
