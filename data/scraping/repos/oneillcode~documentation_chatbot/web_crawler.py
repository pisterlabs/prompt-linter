import requests
import re
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pandas as pd
import tiktoken
import openai
import pinecone


class WebCrawler:

    def __init__(self,
                 embedding_model="text-embedding-ada-002",
                 gpt_model="gpt-3.5-turbo",
                 urls_to_crawl=[],
                 tokenizer_model="cl100k_base",
                 max_tokens=500,
                 debug=False):

        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.urls_to_crawl = urls_to_crawl
        self.max_tokens = max_tokens
        self.debug = debug

        load_dotenv()

        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
                      environment=os.getenv("PINECONE_ENV"))

        # Get API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Load the cl100k_base tokenizer which is designed to work with the emded model
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)

    def remove_newlines(self, serie):
        serie = serie.str.replace('\n', ' ')
        serie = serie.str.replace('\\n', ' ')
        serie = serie.str.replace('  ', ' ')
        serie = serie.str.replace('  ', ' ')
        return serie

    def get_text_between_headers(self, current_text, next_element):
        if next_element.name == 'p':
            current_text.append(next_element.get_text())

        # Check if div with class = 'highlight'
        if next_element.name == 'div' and next_element.get('class') == ['highlight']:
            current_text.append(
                '```' + next_element.get_text() + '```.')

        return current_text

    def get_text_chucks(self, soup_content, title):
        # Soup content is the main content of the page
        # Iterate through the headers and extract the text between them
        headers = soup_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        text_chunks = []

        for i in range(len(headers)):
            header_text = headers[i].get_text()
            next_index = i + 1
            text_chunk = []

            # Find the elements between the current header and the next header (or end of document)
            while next_index < len(headers):
                next_header = headers[next_index]
                next_element = next_header.find_previous()

                # Collect text between the current header and the next header
                current_text = []
                while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    current_text = self.get_text_between_headers(
                        current_text, next_element)

                    next_element = next_element.find_previous()

                # Join together all the text collected
                text_chunk.append(title + " " + header_text +
                                  " " + ' '.join(current_text[::-1]))

                # If text chunk is collected, exit the loop
                if text_chunk:
                    break

                next_index += 1

            # For the last header, collect text from the last header to the end of the document
            if i == len(headers) - 1:
                current_text = []
                next_element = headers[i].find_next()

                while next_element:
                    current_text = self.get_text_between_headers(
                        current_text, next_element)
                    next_element = next_element.find_next()

                text_chunk.append(title + " " + header_text +
                                  " " + ' '.join(current_text))

            # If text chunk is collected, add it to the list
            if text_chunk:
                text_chunks.append({
                    'header': header_text,
                    'text': '\n'.join(text_chunk)
                })

        return text_chunks

    def crawl_url(self, url_to_crawl):
        # Get the text from the URL using BeautifulSoup
        soup = BeautifulSoup(requests.get(url_to_crawl).text, "html.parser")

        # Get title of the page
        title = soup.title.string

        # get only div 'md-content' from the page
        soup_res = soup.find('div', {'class': 'md-content'})

        # if div 'md-content' is not found, get div 'docs-content' as it is a different format of document
        if soup_res is None:
            soup_res = soup.find('div', {'class': 'docs-content'})

        if soup_res is None:
            return None, None

        text_chunks = self.get_text_chucks(soup_res, title)

        # Create a dataframe from the text chucks
        df = pd.DataFrame(text_chunks, columns=['text'])

        # Set the text column to be the raw text with the newlines removed
        df['text'] = self.remove_newlines(df.text)

        # Tokenize the text and save the number of tokens to a new column
        df['n_tokens'] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))

        return df, title

    def split_into_many(self, text):
        # Function to split the text into chunks of a maximum number of tokens

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(self.tokenizer.encode(" " + sentence))
                    for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > self.max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > self.max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        # Add the last chunk to the list of chunks
        if chunk:
            chunks.append(". ".join(chunk) + ".")

        return chunks

    def get_shortened(self, df):
        shortened = []

        # Loop through the dataframe
        for row in df.iterrows():

            # If the text is None, go to the next row
            if row[1]['text'] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]['n_tokens'] > self.max_tokens:
                shortened += self.split_into_many(row[1]['text'])

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append(row[1]['text'])

        df = pd.DataFrame(shortened, columns=['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))

        return df

    def pinecone_get_embeddings(self, df, url, company, title):

        pinecone_embeddings = []
        # Replace space, -, and / with _ and remove all non letters and numbers
        title = title.replace(' ', '_').replace('-', '_').replace('/', '_')
        title = re.sub('[^A-Za-z0-9_]+', '', title)

        df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(
            input=x,
            engine=self.embedding_model
        )['data'][0]['embedding'])

        # loop through the embeddings and add to pinecone index
        for i, row in df.iterrows():
            pinecone_embeddings.append({'id': title + "_" + str(i), 'values': row['embeddings'], 'metadata': {
                                       'url': url, 'text': row['text'], 'company': company}})

        # check if index already exists (only create index if not)
        if 'starburst' not in pinecone.list_indexes():
            pinecone.create_index('starburst', dimension=1536)

        # connect to index
        index = pinecone.Index('starburst')

        index.upsert(pinecone_embeddings)

    def start(self):
        company = 'starburst'

        for url in self.urls_to_crawl:
            df, title = self.crawl_url(url_to_crawl=url)

            if df is None:
                print("ERROR: Unable to parse page " + url)
            else:
                # Shorten the texts to a maximum number of tokens
                df = self.get_shortened(df)

                if self.debug:
                    for i, row in df.iterrows():
                        print(row['text'])
                        print('-------------------')
                        print('-------------------')

                # Get the embeddings for the texts
                self.pinecone_get_embeddings(df, url, company, title)


urls_to_crawl = [
    'https://docs.starburst.io/latest/connector/postgresql.html',
    'https://docs.starburst.io/latest/connector/starburst-snowflake.html',
    'https://docs.starburst.io/latest/connector/starburst-hive.html'
]

web_crawler = WebCrawler(urls_to_crawl=urls_to_crawl, debug=False)

web_crawler.start()
