# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import configparser
import os
import re
import sys

from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

current_dir = os.path.dirname(os.path.abspath(__file__))
spiders_dir = os.path.join(current_dir, 'spiders')

sys.path.append(spiders_dir)

from utils import strip_tags

config = configparser.ConfigParser()
config.read("config.ini")

es = Elasticsearch(
    cloud_id=config["ELASTIC"]["cloud_id"],
    basic_auth=(config["ELASTIC"]["user"], config["ELASTIC"]["password"]),
)


def get_separators_for_language(language: Language) -> list[str]:
    if language == Language.HTML:
        return ["<h1", "<h2", "<h3", "<h4", "<h5", "<h6", ]
    else:
        raise ValueError(
            f"Language {language} is not supported! Please choose from " \
            "{list(Language)}"
        )


def return_splitter(doc_type: str, chunk_size):
    if doc_type == 'html':
        return RecursiveCharacterTextSplitter(
            separators=get_separators_for_language(Language.HTML),
            chunk_size=chunk_size, chunk_overlap=0, keep_separator=True)
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0)


def extract_heading_from_html(chunk: str) -> str:
    soup = BeautifulSoup(chunk, 'html.parser')

    # Find the first heading tag (h1, h2, h3, h4, h5, or h6)
    first_heading_tag = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    if first_heading_tag:
        return first_heading_tag.text.strip()
    else:
        return ''


def extract_bip_number(body_formatted):
    # Regular expression pattern to find the first "<pre>" field containing "BIP"
    pattern = r'<pre>.*?BIP:\s*(\d+).*?<\/pre>'

    # Search for the pattern in the text
    match = re.search(pattern, body_formatted, re.DOTALL)

    if match:
        # Extract the BIP content from the first match
        bip_content = match.group(1).strip()
        return bip_content
    else:
        return None


class ElasticsearchPipeline:
    def process_item(self, item, spider):
        def parse_title(chunk: str) -> str:
            if spider.name in ['andreasbooks', 'btcphilosophy', 'grokkingbtc',
                               'programmingbtc']:
                delim_end = item['title'].find(']')
                return item['title'][1:delim_end] + ':' + \
                    item['title'][delim_end + 1:] + ' - ' + \
                    extract_heading_from_html(chunk)
            elif spider.name == "bips":
                return f"BIP {extract_bip_number(item['body_formatted'])}: " + \
                    item[
                        'title'] + ' - ' + extract_heading_from_html(chunk)
            elif spider.name in ['bitmex', 'bolts']:
                return item['title'] + ' - ' + extract_heading_from_html(chunk)
            elif spider.name == "blips":
                return 'BLIPS: ' + item[
                    'title'] + ' - ' + extract_heading_from_html(chunk)
            elif spider.name == "lndocs":
                return 'LNDocs: ' + item[
                    'title'] + ' - ' + extract_heading_from_html(chunk)
            else:
                return item['title']

        # Split documents for books
        if spider.name in ["bolts", "btcphilosophy", "grokkingbtc", "lndocs",
                           "programmingbtc", "bips", "blips", "andreasbooks",
                           "bitmex"]:
            # Split this first for the body_formatted
            splitter = return_splitter(item['body_type'],
                                       2000)  # 2000 character limit
            chunks = splitter.split_text(item['body_formatted'])

            if len(chunks) > 1:
                for i, chunk in enumerate(chunks):
                    if chunk in ["<article>",
                                 "<article><div>"]:  # Manually filterting btcphilosophy case
                        continue
                    title = parse_title(chunk)
                    body = strip_tags(chunk)
                    document = {**item, 'title': title, 'body_formatted': chunk,
                                'body': body}
                    es.index(index=config["ELASTIC"]["index"],
                             document=document)
            else:
                if spider.name in ['andreasbooks', 'btcphilosophy',
                                   'grokkingbtc',
                                   'programmingbtc']:
                    delim_end = item['title'].find(']')
                    title = item['title']
                    item = {**item, 'title': title[1:delim_end] + ':' + title[
                                                                        delim_end + 1:]}
                es.index(index=config["ELASTIC"]["index"], document=item)
        else:
            es.index(index=config["ELASTIC"]["index"], document=item)
        return item
