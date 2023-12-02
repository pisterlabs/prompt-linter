from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from tqdm import tqdm
from typing import List, Dict
import html2text
from dags.modules.utils.image_ocr import get_image_ocr

from dags.modules.database.pymongo import PymongoClient


class SwconCollector:
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
        #self.documents = self._get_documents(self.links)
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
                f"http://swcon.khu.ac.kr/wordpress/post/?mode=list&board_page={page}"
            )
            soup = BeautifulSoup(response.text, "html.parser")
            for item in soup.select("#DoubleMajor_board_body > tr"):
                link = item.select_one("td.text-left > a").get("href")

                date_text =  item.select("td:nth-child(4) > span")[0].getText() 

                #If the item contains a colon, like 9:45, it means the datetime is today.
                if ":" in date_text:
                    content_date = datetime.today().date()
                else:
                    content_date = datetime.strptime(
                        date_text, "%Y-%m-%d"
                    ).date()
                
                #If the item is a notice, it must continue to do a full scan without stopping.
                if item.select(".mb-notice"):
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
    
    def _extract_img_links(self,html):
        soup = BeautifulSoup(html, 'html.parser')
        img_tags = soup.find_all('img')
        links = []
        for img_tag in img_tags:
            src = img_tag['src']
            if (".jpg" in src) or (".png" in src):
                links.append(img_tag['src'])
        return links
    
    def _transform_documents(self, documents, get_image = True, scope_selector = ".content-box", ignore_links=False, ignore_images=True):
        h = html2text.HTML2Text()
        h.ignore_links = ignore_links
        h.ignore_images = ignore_images

        for document in tqdm(documents):
            page_text = ""
            img_text = ""
            soup = BeautifulSoup(document.page_content, 'html.parser')
            scope_html = str(soup.select_one(scope_selector))
            if get_image:
                img_links = self._extract_img_links(scope_html)
                print(img_links)
                for img_link in img_links:
                    img_text += f" {get_image_ocr(img_link)}"
            page_text = h.handle(scope_html)
            document.page_content = page_text + img_text
        return documents

    def _get_documents(self, links):
        loader = AsyncHtmlLoader(links)
        documents = loader.load()
        transform_documents = self._transform_documents(documents)
        json_documents = self._convert_to_json(transform_documents)
        return json_documents
