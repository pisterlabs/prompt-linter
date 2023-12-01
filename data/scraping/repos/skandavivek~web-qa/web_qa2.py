################################################################################
# Step 1
################################################################################

import os
import re
import openai
import logging
import requests
import tiktoken

import urllib.request
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from openai.embeddings_utils import distances_from_embeddings

# from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

logger = logging.getLogger(__name__)
# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Create a class to parse the HTML and get the hyperlinks


class HyperlinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
# Step 2
################################################################################


def get_hyperlinks(url: str) -> list:
    """ Function to get the hyperlinks from a URL """

    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        logger.error(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
# Step 3
################################################################################


def get_domain_hyperlinks(local_domain: str, url: str) -> list:
    """ Function to get the hyperlinks from a URL that are within the same domain """

    clean_links = []

    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


################################################################################
# Step 4
################################################################################

def crawls(url: str, max_pages: int = 50, hyperlinks: bool = False) -> set:
    """ Function to crawl a website and save the text from each page to a text file """

    # Parse the URL and get the domain
    local_domain = url.replace('/','').replace(':','')

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue and (len(seen) < max_pages):

        # Get the next URL from the queue
        url = queue.pop()
        logger.info(url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding='utf-8') as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                logger.info("Unable to parse page " + url +
                            " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)

        # Get the hyperlinks from the URL and add them to the queue
        if hyperlinks:
            for link in get_domain_hyperlinks(local_domain, url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)

    return seen


################################################################################
# Step 5
################################################################################


def remove_newlines(serie: pd.Series) -> pd.Series:
    """ Function to remove newlines from a series of text """
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
# Step 6
################################################################################

def process(domain: str) -> pd.DataFrame:
    """ Function to process the text files and create a dataframe """

    # Create a list to store the text files
    texts = []

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding='utf-8') as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append(
                (file.replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    # df.head()

    return df

################################################################################
# Step 7
################################################################################


def split_into_many(text: str, tokenizer: tiktoken.Encoding, max_tokens: int = 1024) -> list:
    """ Function to split a string into many strings of a specified number of tokens """

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence))
                for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def tokenize(full_url, api_key: str, max_tokens: int = 1024) -> pd.DataFrame:
    """ Function to split the text into chunks of a maximum number of tokens """

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    if full_url.startswith('http'):
        df = pd.read_csv('processed/scraped.csv', index_col=0)
    else:
        df=pd.DataFrame(['0',full_url]).T
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    # df.n_tokens.hist()

    ################################################################################
    # Step 8
    ################################################################################

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], tokenizer, max_tokens)

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    ################################################################################
    # Step 9
    ################################################################################

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # df.n_tokens.hist()

    ################################################################################
    # Step 10
    ################################################################################

    openai.api_key = api_key
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(
        input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv('processed/embeddings.csv')
    # df.head()

    ################################################################################
    # Step 11
    ################################################################################

    df = pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    # df.head()
    return df

################################################################################
# Step 12
################################################################################


def create_context(question: str, df: pd.DataFrame, max_len: int = 1800, size: str = "ada") -> str:
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def create_context2(question: str, api_key: str, df: pd.DataFrame, max_len: int = 1800, size: str = "ada") -> str:
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    openai.api_key = api_key
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df: pd.DataFrame,
    model: str = "text-davinci-003",
    question: str = "Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len: int = 1800,
    size: str = "ada",
    debug: bool = False,
    max_tokens: int = 150,
    stop_sequence: str = None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        logger.info("Context:" + context)

    try:
        # Create a completions using the questin and context
        # response = openai.Completion.create(
        #     prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        #     temperature=0,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     model=model,
        # )
        # return response["choices"][0]["text"].strip()
        prompt=f"Answer the question. Context: {context}\n Question: {question}"


        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]["message"]["content"]

    except Exception as e:
        logger.error(e)
        return "Sorry, {}".format(str(e))

def answer_question2(
    df: pd.DataFrame,
    api_key: str,
    model: str = "text-davinci-003",
    question: str = "Am I allowed to publish model outputs to Twitter, without a human review?",
    mesg = {},
    max_len: int = 1800,
    size: str = "ada",
    debug: bool = False,
    max_tokens: int = 150,
    stop_sequence: str = None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context2(
        question,
        api_key,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        logger.info("Context:" + context)

    try:
        # Create a completions using the questin and context
        # response = openai.Completion.create(
        #     prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        #     temperature=0,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     model=model,
        # )
        # return response["choices"][0]["text"].strip()
        if(len(mesg)==0):
            prompt=f"Answer the question. Context: {context}\n Question: {question}"

            response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            mesg=[{"role": "user", "content": prompt}]
        else:
            response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            temperature=0,
                messages=mesg
            )

            
        return response['choices'][0]["message"]["content"],mesg

    except Exception as e:
        #logger.error(e)
        return "Sorry, {}".format(str(e))



