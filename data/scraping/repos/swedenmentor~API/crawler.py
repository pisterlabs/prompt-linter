#%% 1.Loading packages
import time                                                         # for time-related tasks
import requests                                                     # for making HTTP requests
import urllib3
import datetime                                                     # for dealing with dates and times
import jsonlines                                                    # for handling JSONL format
import random
import os
from pathlib import Path

from bs4 import BeautifulSoup                                     # for web scraping, parsing HTML
from bs4.element import Tag
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException, ConnectionError, ReadTimeout
from urllib.parse import urlparse, urljoin                          # for URL parsing and joining
from googletrans import Translator                                  # for text translation using Google Translate API
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for splitting text into chunks with overlapping

from utils.custom_logger import CustomLogger
from utils import content_parser as parser


SUPPORTED_LANGUAGES = ['sv', 'en']


class WriteMode():
    APPEND = 'a'
    OVERWRITE = 'w'

def write_jsonl(input_entries: list[dict], output_file: str, write_mode: WriteMode):
    """Write content to jsonl format

    Parameters
    ----------
    input_entries : list[dict]
    output_file : str
    mode : WriteMode
        - 'APPEND' to append new content to the file
        - 'OVERWRITE' to overwrite content of the file
    """

    folder = Path(output_file).parent

    if not folder.exists():
        os.makedirs(folder)

    with jsonlines.open(output_file, write_mode) as f:
        for entry in input_entries:
            f.write(entry)


#%% 2.Define function to crawl data from the website
class Crawler:
    """Master Crawler class
    """
    def __init__(
        self, 
        logger: CustomLogger = None
    ):
        
        self.target_urls = set()    # Set of URLs to be crawled
        self.visited_urls = set()   # Set of URLs visited
        self.translator = Translator()
        self.splitter: RecursiveCharacterTextSplitter = None    # Placeholder for the splitter object
        self.data_buffer = []
        self.logger = logger or CustomLogger(name=self.__class__.__name__, write_local=False) 
        
        self.session = self.get_session()
    
    def get_session(
        self,
        total_retries: int = 3, 
        backoff_factor: float = 0.1, 
        status_forcelist: list[int] = [500, 502, 503, 504, 429],
    ):
        """Generate a session object with retry settings.

        Parameters
        ----------
        total_retries : int
            Total number of retries allowed
        backoff_factor : float
            This parameter affects how long the process waits before retrying a request.
            wait_time = {backoff factor} * (2 ** ({number of total retries} - 1))
            For example, if the backoff_factor is 0.1, the process will sleep for [0.1s, 0.2s, 0.4s, ...] between retries.
        status_forcelist : list[int]
            List of status codes that will trigger a retry.
        """
        retries = urllib3.Retry(
            total=total_retries, 
            backoff_factor=backoff_factor, 
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session = requests.Session()
        session.mount('http://', adapter)

        return session
        
    def get_url(self, url: str, timeout: int = 300): 
        """Get the response from a URL.

        Parameters
        ----------
        url : str
            The URL to get the response from.
        """
        
        response = self.session.get(url, timeout=timeout)
        try:
            response.raise_for_status()
        except (HTTPError, RequestException, ConnectionError, ReadTimeout) as err:
            self.logger.error(f'Error when connecting: {err}')
        else:
            return response

    def translate_text(self, text: str, retry_limit: int = 3) -> str:
        """Translate the input text using Translator() object.
        Return original text if the language is en

        Parameters
        ----------
        text : str
        retry_limit : int, optional
            Number of retries allowed, by default 3

        Returns
        -------
        str

        Raises
        ------
        AttributeError
            Raise error if we cannot translate the text
        """

        retry_count = 0
        detected = self.translator.detect(text[:1000])
        translated_text = None

        if detected.lang != 'en':
            return text
        
        # Retry until the translation is successful, or reach retry limit
        while retry_count < retry_limit:
            retry_count += 1
            try:
                translated_text = self.translator.translate(text, src=detected.lang, dest='en').text
                break
            except AttributeError as e:
                self.logger.error(f"Error during translation: {e}, attempt {retry_count}. Retrying...")
                time.sleep(2)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during translation: {e}")
                raise e

        return translated_text

    def check_language_supported(self, soup: BeautifulSoup) -> bool:
        """Check if the page is written in a supported language (Defined by the SUPPORTED_LANGUAGE variable)

        Parameters
        ----------
        soup : BeautifulSoup
            A BeautifulSoup object representing the text content

        Returns
        -------
        bool
        """
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        check_lang = "\n".join([parser.clean_text(p.get_text()) for p in paragraphs])[:1000]

        detected = self.translator.detect(check_lang)
        
        if detected.lang not in SUPPORTED_LANGUAGES:
            self.logger.warning(f'The page is written in an unsupported language: {detected.lang}')
            return False
        else:
            return True

    def crawl_links(
        self, 
        url: str, 
        depth: int = 10, 
        base_url: str = '',
        exclude_urls: list[str] = None
    ):
        """Recursively crawl links in a web site, and return a list of URLs.

        Parameters
        ----------
        url : str
        depth : int, optional
            The depth that we want to go, by default 10 pages.
        base_url : str, optional
            The base url to be used if the href attributes are relative path instead of full URL.
        exclude_urls : list[str], optional
            Specify a list of URLs to be excluded from the results
        """

        # Check if the depth is 0 or the url has been visited
        if url in self.target_urls:
            self.logger.info("URL existed - skip.")
            return
        if depth == 0:
            self.logger.info("Reached travel depth.")
            return

        if exclude_urls:
            match_excluded = [url.startswith(exc) for exc in exclude_urls]
            if any(match_excluded):
                return

        self.logger.info(f'Visiting: {url}')
        response = self.get_url(url)
        self.target_urls.add(url)  # Add the URL to the result set if it is valid

        # Check if the website has supported languages
        soup = BeautifulSoup(response.text, 'html.parser')

        if self.check_language_supported(soup) is False:
            return

        # Use recursion to find all links
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')

            if (href is None) or (len(href) == 0) or ('#' in href): 
                continue

            exc_patterns = ('.xml', '.pdf', '.jpg', '.png', '.zip', '.printable', '.contenttype=text/xml;charset=UTF-8')
            if href.lower().endswith(exc_patterns): 
                continue

            if href and href.startswith('http'):
                new_url = href
            else:
                new_url = urljoin(base_url, href)
            
            if urlparse(new_url).netloc == urlparse(url).netloc and new_url not in self.visited_urls:
                time.sleep(random.randint(0, 10) / 100)  
                self.crawl_links(new_url, depth=depth - 1)

    def write_urls(
        self, 
        url_list: list[str], 
        output_file: str, 
        write_mode: WriteMode = WriteMode.OVERWRITE
    ):
        """Function to write the list of URLs to a file

        Parameters
        ----------
        url_list : list[str]
        output_file : str
        """
        folder = Path(output_file).parent

        if not folder.exists():
            os.makedirs(folder)

        with open(output_file, mode=write_mode, encoding='utf-8') as f:
            for url in url_list:
                f.write(f"{url}\n")

    def extract_json(
            self, 
            base_url: str, 
            output_file: str = None
    ):
        """Extract content from response.json() instead of response's text.

        Parameters
        ----------
        base_url : str
            Base URL to iterate through
        output_file : str, optional
            Path to the jsonl file to write the full parsed text to, by default None
        """
        page = 1
            
        while True:
            url = base_url.format(page)
            response = self.get_url(url)

            # If the response status code is not 200, break the loop
            if response.status_code != 200:
                break

            # Check if the url has been visited
            if url in self.visited_urls:
                break
            
            # Extract the JSON data from the response
            response_json = response.json()

            buffer = []
            for item in response_json:

                # Safely get contents from json
                title = item.get('title', {}).get('rendered')
                rendered_content = item.get('content', {}).get('rendered')
                full_text = parser.clean_text(BeautifulSoup(rendered_content, 'html.parser').get_text(separator='').strip())
                sub_url = item.get('link')
                updated_date = datetime.datetime.strptime(item['date'], "%Y-%m-%dT%H:%M:%S") \
                    .strftime("%Y-%m-%d") if 'date' in item else None

                result = {
                    'source': url,
                    'title': title,
                    'full_text': full_text,
                    'updated': updated_date
                }

                buffer.append(result)

            self.data_buffer += buffer

            if output_file:
                write_jsonl(buffer, output_file, write_mode=WriteMode.APPEND)

            # Increase the page number for the next iteration
            self.visited_urls.add(sub_url)
            page += 1
        

    def extract_web_element(
        self, 
        input_file: str = None, 
        output_file: str = None,
        scope_selector: str = '',
        target_tags: list[str] = ['p', 'h1', 'h2'], 
        start: int = 0, 
        end: int = None, 
        special_tags: list[str] = None, 
        class_name: str = None,
        base_url: str = '',
        write_mode: WriteMode = WriteMode.APPEND
    ):
        """Iterate through URLs, parse text, and add to self.data_buffer.
        Write to local file if specified.

        Parameters
        ----------
        input_file : str, optional
            Path to a text file containing URLs to be crawled, by default None
        output_file : str, optional
            Path to a .jsonl file to write the outpot to, by default None
        scope_selector : str, optional
            If specifying a CSS selector, the crawler will find target_tags within the selected scope.
        target_tags : list[str], optional
            List of tags to be targeted, by default ['p', 'h1', 'h2']
        start : int, optional
            Starting point of the URLs list to be crawled, by default 0
        end : int, optional
            Ending point of the URLs list to be crawled, by default None
        special_tags : list[str], optional
            If special tags is specified, only get these tags that has the specified class_name , by default None
        class_name : str, optional
            This is to be used in conjunction with special_tags argument, by default None
        base_url : str, optional
            Base URL to append to relative paths crawled in the website, if any. By default ''
        write_mode : WriteMode
        """

        if input_file is not None:
            with open(input_file, 'r', encoding='utf-8') as f:
                target_urls = [line.strip() for line in f]
        else:
            target_urls = list(self.target_urls)

        buffer = []

        for url in target_urls[start:end]:
            self.logger.info(f'Visiting: {url}')
            response = self.get_url(url)

            # Extract web elements
            master_soup = BeautifulSoup(response.text, 'html.parser')

            if scope_selector:
                soup = master_soup.select(scope_selector)[0]
            else:
                soup = master_soup

            target_elms: list[Tag] = soup.find_all(target_tags)
            
            paragraph_texts = []

            for elm in target_elms:
                
                # If encountering a special tag, only get the tag with the specified class name
                if special_tags is not None and class_name is not None:
                    if elm.name in special_tags and elm.get('class', [None]) != [class_name]:
                        continue
                
                text = parser.parse_content(elm, base_url=base_url)
                paragraph_texts.append(text)

            full_text = "\n".join(paragraph_texts)

            # TODO: Find all date tags for all websites
            date_tag = soup.find('p', class_='ahjalpfunktioner')
            if date_tag:
                time_tag = date_tag.find('time')
                updated_date = time_tag.get_text() if time_tag else datetime.datetime.now().strftime("%Y-%m-%d")
            else:
                updated_date = datetime.datetime.now().strftime("%Y-%m-%d")

            # Return result as a dict
            title = master_soup.title.text.strip()

            result = {
                'source': url,
                'title': title,
                'full_text': full_text,
                'updated': updated_date
            }

            buffer.append(result)

            # Append this url's result set to self.data_buffer
            self.visited_urls.add(url)
        
        self.data_buffer += buffer

        # Write result to file if specified
        if output_file:
            self.logger.info(f"Write self.data_buffer to file {output_file}")
            write_jsonl(self.data_buffer, output_file, write_mode=write_mode)

    def write_chunk_text(
            self, 
            input_file: str = None,
            output_file: str = None, 
            translate: bool = False,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            write_mode: WriteMode = WriteMode.APPEND
    ):
        """Iterate through parsed text in self.data_buffer, break text into chunk, and write to file.

        Parameters
        ----------
        input_file : str 
            Path to the input jsonl file containing the full text
        output_file : str
            Path to a jsonl file to write the chunks to
        translate : bool, optional
            Specify if the text should be translated, by default False
        chunk_size : int, optional
            Maximum number of characters in the chunk, by default 1000
        chunk_overlap : int, optional
            Number of overlap characters between two adjacent chunks, by default 200
        """
        if self.splitter is None:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
        
        if input_file:
            with jsonlines.open(input_file, 'r') as f:
                data_buffer = list(f.iter())
                self.data_buffer = data_buffer

        chunk_buffer = []

        for entry in self.data_buffer:
            self.logger.info(f"Chunking {entry['source']}")
            if translate:
                full_text = self.translate_text(entry['full_text'])
                title = self.translate_text(entry['title'])
            else:
                full_text = entry['full_text']
                title = entry['title']
            
            for idx, chunk in enumerate(self.splitter.split_text(full_text)):
                chunk_out = {
                    "chunk-id": str(idx),
                    "source": entry['source'],
                    "title": title,
                    "chunk": chunk,
                    "updated": entry['updated'],
                }
                chunk_buffer.append(chunk_out)

        self.logger.info(f"Write text chunks to {output_file}")
        write_jsonl(chunk_buffer, output_file, write_mode=write_mode)
