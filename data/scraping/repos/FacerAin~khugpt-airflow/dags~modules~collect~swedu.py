from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from tqdm import tqdm
from typing import List, Dict

from dags.modules.database.pymongo import PymongoClient


class SweduCollector:
    today_date = date.today()
    tomarrow_date = date.today() + timedelta(days=1)

    def __init__(self):
        self.links = []
        self.documents = []

    def collect(
        self,
        start_date: datetime.date = today_date,
        end_date: datetime.date = tomarrow_date,
    ):
        # TODO: Serve progress informations.
        self.links = self._get_links(start_date, end_date)
        self.documents = self._get_documents(self.links)
        return self.documents

    def upload_db(self, db_host="localhost", db_port=27017):
        client = PymongoClient(host=db_host, port=db_port)
        client.insert_documents(self.documents)

    def _convert_to_json(self, documents):
        items = []
        for document in documents:
            items.append(
                {
                    "page_content": document.page_content,
                    "page_url": document.metadata["source"],
                    "collected_at": str(date.today()),
                }
            )

        return items
    
    def _clean_json_documents(self, json_documents: List[Dict]):
        clean_json_documents = []
        for document in json_documents:
            document['page_content'] = self._clean_page_content(document['page_content'])
            clean_json_documents.append(document
        )
        return clean_json_documents
    
    def _clean_page_content(self, page_content: str):
        # Remove the header section based on the "## 공지사항"
        clean_page_content = ' '.join(page_content.split("## 공지사항")[1:])

        # Remove the footer section based on the "* __다음글"
        clean_page_content = ' '.join(clean_page_content.split("* __목록")[:-1])

        return clean_page_content


    def _get_links(self, start_date, end_date, max_page: int = 34):
        # TODO: How to determine max page?
        links = []
        is_break = False
        for page in tqdm(range(1, max_page)):
            if is_break:
                break
            response = requests.get(
                f"https://swedu.khu.ac.kr/board5/bbs/board.php?bo_table=06_01&page={page}"
            )
            soup = BeautifulSoup(response.text, "html.parser")
            for item in soup.select("#fboardlist > div > table > tbody > tr"):
                link = item.find("a").get("href")
                content_date = datetime.strptime(
                    item.select(".td_datetime")[0].getText(), "%Y-%m-%d"
                ).date()
                
                #If the item is a notice, it must continue to do a full scan without stopping.
                if item.select(".notice_icon"):
                    if  start_date <= content_date<= end_date:
                        links.append(link)
                else:
                    if end_date < content_date:
                        continue
                    elif start_date > content_date:
                        is_break = True
                        break
                    else:
                        links.append(link)
        return links

    def _get_documents(self, links):
        loader = AsyncHtmlLoader(links)
        html2text = Html2TextTransformer()
        documents = loader.load()
        transform_documents = html2text.transform_documents(documents)
        json_documents = self._convert_to_json(transform_documents)
        json_documents = self._clean_json_documents(json_documents)
        return json_documents
