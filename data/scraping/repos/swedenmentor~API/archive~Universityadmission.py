import time
import requests
from bs4 import BeautifulSoup
import datetime
import jsonlines
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Crawler_UA:
    def __init__(self):
        self.visited_urls = set()
        self.translator = Translator()
        self.data_buffer = []

    def write_to_file(self, output_file):
        """
        Write data from data buffer to a file.

        :param output_file: The path to the file where the data will be written.
        :type output_file: str
        :return: None
        :rtype: None
        """
        if self.data_buffer:
            with jsonlines.open(output_file, mode='a') as file:
                for entry in self.data_buffer:
                    file.write(entry)
                self.data_buffer = []

    def crawl_study_programs(self, output_file):
        base_url = 'https://www.universityadmissions.se/intl/search?period=21&sortBy=nameAsc&page='
        max_pages = 32

        for page in range(1, max_pages + 1):
            url = f'{base_url}{page}'
            print('Processing ' + url)

            response = requests.get(url)
            if not response.status_code == 200:
                print(f'Non-success status for url {url}')
                return

            soup = BeautifulSoup(response.text, 'html.parser')
            resultsection_div = soup.find('div', {"class": 'resultsection'})
            searchresultcards = resultsection_div.find_all('div', {"class": 'searchresultcard'})

            for idx, card in enumerate(searchresultcards):
                chunk_list = []
                # Program name extraction
                program_name = card.find('h3', {"class": 'headline4'}).text
                chunk_list += [program_name]
                chunk_list += card.find('p', {"class": 'universal_medium'}).text.strip().split('\n')
                # Detail information, credits, degree, location and subject areas.
                detail_card = card.find('div', {"class": 'resultcard_expanded universal_high'}).find_all('p')
                details = [p.get_text().replace('\n', ' ').rstrip() for p in detail_card]
                chunk_list += details
                # Extract url to the university
                link = card.find('div', {"class": 'resultcard_expanded universal_high'}).find_all('a')[0].get('href')
                chunk_list += [link]

                chunk = ' '.join(chunk_list)

                entries = {
                    "chunk-id": str(idx),
                    "source": link,
                    "title": program_name,
                    "chunk": chunk,
                    "updated": datetime.datetime.now().strftime("%Y-%m-%d"),
                }

                self.data_buffer.append(entries)
                self.write_to_file(output_file)

if __name__ == '__main__':
    crawler = Crawler_UA()
    output_file = 'universityadmissions.jsonl'
    
    # Clear existing content in the JSONL file
    with open(output_file, 'w'):pass

    crawler.crawl_study_programs(output_file=output_file)
