import re
import pandas as pd
from feedgen.feed import FeedGenerator
from tqdm import tqdm
from elasticsearch import Elasticsearch
import time
import traceback
import platform
import openai
import shutil
from datetime import datetime, timedelta
import pytz
import glob
import xml.etree.ElementTree as ET
import tiktoken
import os
from dotenv import load_dotenv
import sys
import ast
from loguru import logger
import warnings
from openai.error import APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError
from src.utils import preprocess_email
from src.gpt_utils import generate_chatgpt_summary, consolidate_chatgpt_summary
from src.config import TOKENIZER, ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX, ES_DATA_FETCH_SIZE

warnings.filterwarnings("ignore")
load_dotenv()


TOKENIZER = tiktoken.get_encoding("cl100k_base")

# if set to True, it will use chatgpt model ("gpt-4-1106-preview") for all the completions
CHATGPT = True

# COMPLETION_MODEL - only applicable if CHATGPT is set to False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY


class ElasticSearchClient:
    def __init__(self, es_cloud_id, es_username, es_password, es_data_fetch_size=ES_DATA_FETCH_SIZE) -> None:
        self._es_cloud_id = es_cloud_id
        self._es_username = es_username
        self._es_password = es_password
        self._es_data_fetch_size = es_data_fetch_size
        self._es_client = Elasticsearch(
            cloud_id=self._es_cloud_id,
            http_auth=(self._es_username, self._es_password),
        )

    def extract_data_from_es(self, es_index, url, start_date_str, current_date_str):
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "prefix": {  # Using prefix query for domain matching
                                    "domain.keyword": str(url)
                                }
                            },
                            {
                                "range": {
                                    "created_at": {
                                        "gte": f"{start_date_str}T00:00:00.000Z",
                                        "lte": f"{current_date_str}T23:59:59.999Z"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Initialize the scroll
            scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
                                                     scroll='1m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='1m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return None


class GenerateXML:
    def __init__(self) -> None:
        self.month_dict = {
            1: "Jan", 2: "Feb", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
        }

    def split_prompt_into_chunks(self, prompt, chunk_size):
        tokens = TOKENIZER.encode(prompt)
        chunks = []

        while len(tokens) > 0:
            current_chunk = TOKENIZER.decode(tokens[:chunk_size]).strip()

            if current_chunk:
                chunks.append(current_chunk)

            tokens = tokens[chunk_size:]

        return chunks

    def get_summary_chunks(self, body, tokens_per_sub_body):
        chunks = self.split_prompt_into_chunks(body, tokens_per_sub_body)
        summaries = []

        logger.info(f"Total chunks: {len(chunks)}")

        for chunk in chunks:
            count_gen_sum = 0
            while True:
                try:
                    time.sleep(2)
                    summary = generate_chatgpt_summary(chunk)
                    summaries.append(summary)
                    break
                except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
                    logger.error(str(ex))
                    count_gen_sum += 1
                    time.sleep(0.2)
                    if count_gen_sum > 5:
                        sys.exit(f"Chunk summary ran into error: {traceback.format_exc()}")

        return summaries

    def recursive_summary(self, body, tokens_per_sub_body, max_length):
        summaries = self.get_summary_chunks(body, tokens_per_sub_body)

        summary_length = sum([len(TOKENIZER.encode(s)) for s in summaries])

        logger.info(f"Summary length: {summary_length}")
        logger.info(f"Max length: {max_length}")

        if summary_length > max_length:
            logger.info("entering in recursion")
            return self.recursive_summary("".join(summaries), tokens_per_sub_body, max_length)
        else:
            return summaries

    def gpt_api(self, body):
        body_length_limit = 2800
        tokens_per_sub_body = 2700
        summaries = self.recursive_summary(body, tokens_per_sub_body, body_length_limit)

        if len(summaries) > 1:
            logger.info("Consolidate summary generating")
            summary_str = "\n".join(summaries)
            count_api = 0
            while True:
                try:
                    time.sleep(2)
                    consolidated_summaries = consolidate_chatgpt_summary(summary_str)
                    break
                except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
                    logger.error(str(ex))
                    count_api += 1
                    time.sleep(0.2)
                    if count_api > 5:
                        sys.exit(f"Chunk summary ran into error: {traceback.format_exc()}")

            return consolidated_summaries

        else:
            logger.info("Individual summary generating")
            return "\n".join(summaries)

    def create_summary(self, body):
        summ = self.gpt_api(body)
        return summ

    def create_folder(self, month_year):
        os.makedirs(month_year, exist_ok=True)

    def generate_xml(self, feed_data, xml_file):
        # create feed generator
        fg = FeedGenerator()
        fg.id(feed_data['id'])
        fg.title(feed_data['title'])
        for author in feed_data['authors']:
            fg.author({'name': author})
        for link in feed_data['links']:
            fg.link(href=link, rel='alternate')
        # add entries to the feed
        fe = fg.add_entry()
        fe.id(feed_data['id'])
        fe.title(feed_data['title'])
        fe.link(href=feed_data['url'], rel='alternate')
        fe.published(feed_data['created_at'])
        fe.summary(feed_data['summary'])

        # generate the feed XML
        feed_xml = fg.atom_str(pretty=True)
        # convert the feed to an XML string
        # write the XML string to a file
        with open(xml_file, 'wb') as f:
            f.write(feed_xml)

    def clean_title(self, xml_name):
        special_characters = ['/', ':', '@', '#', '$', '*', '&', '<', '>', '\\', '?']
        xml_name = re.sub(r'[^A-Za-z0-9]+', '-', xml_name)
        for sc in special_characters:
            xml_name = xml_name.replace(sc, "-")
        return xml_name

    def remove_multiple_whitespaces(self, text):
        return re.sub('\s+', ' ', text).strip()

    def get_id(self, id):
        return str(id).split("-")[-1]

    def append_columns(self, df_dict, file, title, namespace):
        df_dict["body_type"].append(0)
        df_dict["id"].append(file.split("/")[-1].split("_")[0])
        df_dict["type"].append(0)
        df_dict["_index"].append(0)
        df_dict["_id"].append(0)
        df_dict["_score"].append(0)

        df_dict["title"].append(title)
        formatted_file_name = file.split("/static")[1]
        logger.info(formatted_file_name)

        tree = ET.parse(file)
        root = tree.getroot()

        date = root.find('atom:entry/atom:published', namespace).text
        datetime_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+00:00")
        timezone = pytz.UTC
        datetime_obj = datetime_obj.replace(tzinfo=timezone)
        df_dict["created_at"].append(datetime_obj)

        link_element = root.find('atom:entry/atom:link', namespace)
        link_href = link_element.get('href')
        df_dict["url"].append(link_href)

        author = root.find('atom:author/atom:name', namespace).text
        author_result = re.sub(r"\d", "", author)
        author_result = author_result.replace(":", "")
        author_result = author_result.replace("-", "")
        df_dict["authors"].append([author_result.strip()])

    def file_not_present_df(self, columns, source_cols, df_dict, files_list, dict_data, data,
                            title, combined_filename, namespace):
        for col in columns:
            df_dict[col].append(dict_data[data][col])

        for col in source_cols:
            if "created_at" in col:
                datetime_obj = datetime.strptime(dict_data[data]['_source'][col], "%Y-%m-%dT%H:%M:%S.%fZ")
                timezone = pytz.UTC
                datetime_obj = datetime_obj.replace(tzinfo=timezone)
                df_dict[col].append(datetime_obj)
            else:
                df_dict[col].append(dict_data[data]['_source'][col])
        for file in files_list:
            file = file.replace("\\", "/")
            if os.path.exists(file):
                tree = ET.parse(file)
                root = tree.getroot()
                file_title = root.find('atom:entry/atom:title', namespace).text

                if title == file_title:
                    self.append_columns(df_dict, file, title, namespace)

                    if combined_filename in file:
                        tree = ET.parse(file)
                        root = tree.getroot()
                        summary = root.find('atom:entry/atom:summary', namespace).text
                        df_dict["body"].append(summary)
                    else:
                        summary = root.find('atom:entry/atom:summary', namespace).text
                        df_dict["body"].append(summary)
            else:
                logger.info(f"File not present:- {file}")

    def file_present_df(self, files_list, namespace, combined_filename, title, xmls_list, df_dict):
        combined_file_fullpath = None
        month_folders = []
        for file in files_list:
            file = file.replace("\\", "/")
            if combined_filename in file:
                combined_file_fullpath = file
            tree = ET.parse(file)
            root = tree.getroot()
            file_title = root.find('atom:entry/atom:title', namespace).text
            if title == file_title:
                xmls_list.append(file)
                month_folder_path = "/".join(file.split("/")[:-1])
                if month_folder_path not in month_folders:
                    month_folders.append(month_folder_path)

        for month_folder in month_folders:
            if combined_file_fullpath and combined_filename not in os.listdir(month_folder):
                if combined_filename not in os.listdir(month_folder):
                    shutil.copy(combined_file_fullpath, month_folder)

        if len(xmls_list) > 0 and not any(combined_filename in item for item in files_list):
            logger.info("individual summaries are present but not combined")
            for file in xmls_list:
                self.append_columns(df_dict, file, title, namespace)
                tree = ET.parse(file)
                root = tree.getroot()
                summary = root.find('atom:entry/atom:summary', namespace).text
                df_dict["body"].append(summary)

    def convert_to_tuple(self, x):
        try:
            if isinstance(x, str):
                x = ast.literal_eval(x)
            return tuple(x)
        except ValueError:
            return (x,)

    def preprocess_authors_name(self, author_tuple):
        author_tuple = tuple(s.replace('+', '').strip() for s in author_tuple)
        return author_tuple

    def add_utc_if_not_present(self, datetime_str):
        try:
            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S%z")
            datetime_obj = datetime_obj.replace(tzinfo=pytz.UTC)
        except ValueError:
            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
            timezone = pytz.UTC
            datetime_obj = datetime_obj.replace(tzinfo=timezone)
        return datetime_obj.isoformat(" ")

    def generate_new_emails_df(self, dict_data, dev_url):
        columns = ['_index', '_id', '_score']
        source_cols = ['body_type', 'created_at', 'id', 'title', 'body', 'type',
                       'url', 'authors']
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}

        current_directory = os.getcwd()

        if "lightning-dev" in dev_url:
            files_list = glob.glob(os.path.join(current_directory, "static", "lightning-dev", "**/*.xml"), recursive=True)
        else:
            files_list = glob.glob(os.path.join(current_directory, "static", "bitcoin-dev", "**/*.xml"), recursive=True)

        df_dict = {}
        for col in columns:
            df_dict[col] = []
        for col in source_cols:
            df_dict[col] = []

        for data in range(len(dict_data)):
            xmls_list = []
            number = self.get_id(dict_data[data]["_source"]["id"])
            title = dict_data[data]["_source"]["title"]
            xml_name = self.clean_title(title)
            file_name = f"{number}_{xml_name}.xml"
            combined_filename = f"combined_{xml_name}.xml"

            if not any(file_name in item for item in files_list):
                logger.info(f"{file_name} is not present")
                self.file_not_present_df(columns, source_cols, df_dict, files_list, dict_data, data,
                                         title, combined_filename, namespace)

            else:
                logger.info(f"{file_name} already exist")
                self.file_present_df(files_list, namespace, combined_filename, title, xmls_list, df_dict)

        emails_df = pd.DataFrame(df_dict)

        emails_df['authors'] = emails_df['authors'].apply(self.convert_to_tuple)
        emails_df = emails_df.drop_duplicates()
        emails_df['authors'] = emails_df['authors'].apply(self.preprocess_authors_name)
        emails_df['body'] = emails_df['body'].apply(preprocess_email)
        emails_df['title'] = emails_df['title'].apply(self.remove_multiple_whitespaces)
        logger.info(f"Shape of emails_df: {emails_df.shape}")
        return emails_df

    def start(self, dict_data, url):
        if len(dict_data) > 0:
            emails_df = self.generate_new_emails_df(dict_data, url)
            if len(emails_df) > 0:
                emails_df['created_at_org'] = emails_df['created_at'].astype(str)

                def generate_local_xml(cols, combine_flag, url):
                    month_name = self.month_dict[int(cols['created_at'].month)]
                    str_month_year = f"{month_name}_{int(cols['created_at'].year)}"

                    if "bitcoin-dev" in url:
                        if not os.path.exists(f"static/bitcoin-dev/{str_month_year}"):
                            self.create_folder(f"static/bitcoin-dev/{str_month_year}")
                        number = self.get_id(cols['id'])
                        xml_name = self.clean_title(cols['title'])
                        file_path = f"static/bitcoin-dev/{str_month_year}/{number}_{xml_name}.xml"
                    else:
                        if not os.path.exists(f"static/lightning-dev/{str_month_year}"):
                            self.create_folder(f"static/lightning-dev/{str_month_year}")
                        number = self.get_id(cols['id'])
                        xml_name = self.clean_title(cols['title'])
                        file_path = f"static/lightning-dev/{str_month_year}/{number}_{xml_name}.xml"
                    if os.path.exists(file_path):
                        logger.info(f"{file_path} already exist")
                        if "bitcoin-dev" in url:
                            link = f'bitcoin-dev/{str_month_year}/{number}_{xml_name}.xml'
                        else:
                            link = f'lightning-dev/{str_month_year}/{number}_{xml_name}.xml'
                        return link
                    summary = self.create_summary(cols['body'])
                    feed_data = {
                        'id': combine_flag,
                        'title': cols['title'],
                        'authors': [f"{cols['authors'][0]} {cols['created_at']}"],
                        'url': cols['url'],
                        'links': [],
                        'created_at': cols['created_at_org'],
                        'summary': summary
                    }
                    self.generate_xml(feed_data, file_path)
                    if "bitcoin-dev" in url:
                        link = f'bitcoin-dev/{str_month_year}/{number}_{xml_name}.xml'
                    else:
                        link = f'lightning-dev/{str_month_year}/{number}_{xml_name}.xml'
                    return link

                # combine_summary_xml
                os_name = platform.system()
                logger.info(f"Operating System: {os_name}")
                titles = emails_df.sort_values('created_at')['title'].unique()
                logger.info(f"Total titles in data: {len(titles)}")
                for title_idx, title in tqdm(enumerate(titles)):
                    title_df = emails_df[emails_df['title'] == title]
                    title_df['authors'] = title_df['authors'].apply(self.convert_to_tuple)
                    title_df = title_df.drop_duplicates()
                    title_df['authors'] = title_df['authors'].apply(self.preprocess_authors_name)
                    title_df = title_df.sort_values(by='created_at', ascending=False)
                    logger.info(f"length of title_df: {len(title_df)}")
                    if len(title_df) < 1:
                        continue
                    if len(title_df) == 1:
                        generate_local_xml(title_df.iloc[0], "0", url)
                        continue
                    # body = title_df['body'].apply(str).tolist() + old_files_data_dict["summary_list"]
                    combined_body = '\n\n'.join(title_df['body'].apply(str))
                    # combined_body = '\n\n'.join(body)
                    xml_name = self.clean_title(title)
                    combined_links = list(title_df.apply(generate_local_xml, args=("1", url), axis=1))
                    combined_authors = list(
                        title_df.apply(lambda x: f"{x['authors'][0]} {x['created_at']}", axis=1))

                    month_year_group = \
                        title_df.groupby([title_df['created_at'].dt.month, title_df['created_at'].dt.year])

                    flag = False
                    std_file_path = ''
                    for idx, (month_year, _) in enumerate(month_year_group):
                        logger.info(f"###### {month_year}")
                        month_name = self.month_dict[int(month_year[0])]
                        str_month_year = f"{month_name}_{month_year[1]}"
                        if "bitcoin-dev" in url:
                            file_path = f"static/bitcoin-dev/{str_month_year}/combined_{xml_name}.xml"
                        else:
                            file_path = f"static/lightning-dev/{str_month_year}/combined_{xml_name}.xml"
                        combined_summary = self.create_summary(combined_body)
                        feed_data = {
                            'id': "2",
                            'title': 'Combined summary - ' + title,
                            'authors': combined_authors,
                            'url': title_df.iloc[0]['url'],
                            'links': combined_links,
                            'created_at': self.add_utc_if_not_present(title_df.iloc[0]['created_at_org']),
                            'summary': combined_summary
                        }
                        if not flag:
                            self.generate_xml(feed_data, file_path)
                            std_file_path = file_path
                            flag = True
                        else:
                            if os_name == "Windows":
                                shutil.copy(std_file_path, file_path)
                            elif os_name == "Linux":
                                os.system(f"cp {std_file_path} {file_path}")
            else:
                logger.info("No new files are found")
        else:
            logger.info("No input data found")


if __name__ == "__main__":
    gen = GenerateXML()
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)
    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/"
    ]

    current_date_str = None
    if not current_date_str:
        current_date_str = datetime.now().strftime("%Y-%m-%d")

    start_date = datetime.now() - timedelta(days=30)
    start_date_str = start_date.strftime("%Y-%m-%d")
    logger.info(f"start_data: {start_date_str}")
    logger.info(f"current_date_str: {current_date_str}")

    for dev_url in dev_urls:
        data_list = elastic_search.extract_data_from_es(ES_INDEX, dev_url, start_date_str, current_date_str)
        dev_name = dev_url.split("/")[-2]
        logger.info(f"Total threads received for {dev_name}: {len(data_list)}")

        delay = 5
        count_main = 0

        while True:
            try:
                gen.start(data_list, dev_url)
                break
            except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
                logger.error(str(ex))
                logger.error(ex)
                time.sleep(delay)
                count_main += 1
                if count_main > 5:
                    sys.exit(ex)
