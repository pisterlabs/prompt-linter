# This code scrapes a website, splits the text into chunks, and embeds them using OpenAI and Pinecone.

from time import sleep
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os
import re
import html
import urllib.parse
from bs4 import BeautifulSoup
import requests
import openai
import pinecone
from bs4 import BeautifulSoup
import numpy as np
import sys
import streamlit as st
from mojafunkcja import pinecone_stats, st_style, tiktoken_len


st_style()


def scrape(url: str):
    global headers, sajt, err_log, tiktoken_len, vrsta
    # Send a GET request to the URL
    res = requests.get(url, headers=headers)

    # Check the response status code
    if res.status_code != 200:
        # If the status code is not 200 (OK), write the status code and return None
        err_log += f"{res.status_code} for {url}\n"
        return None

    # If the status code is 200, initialize BeautifulSoup with the response text
    soup = BeautifulSoup(res.text, 'html.parser')
    # soup = BeautifulSoup(res.text, 'lxml')

    # Find all links to local pages on the website
    local_links = []
    for link in soup.find_all('a', href=True):
        if link['href'].startswith(sajt) or link['href'].startswith('/') or link['href'].startswith('./'):
            href = link['href']
            base_url, extension = os.path.splitext(href)
            if not extension and not "mailto" in href and not "tel" in href:
                local_links.append(
                    urllib.parse.urljoin(sajt, href))

    # Find the main content using CSS selectors
                try:
                    # main_content_list = soup.select('body main')
                    main_content_list = soup.select(vrsta)

                    # Check if 'main_content_list' is not empty
                    if main_content_list:
                        main_content = main_content_list[0]

                        # Extract the plaintext of the main content
                        main_content_text = main_content.get_text()

                        # Remove all HTML tags
                        main_content_text = re.sub(
                            r'<[^>]+>', '', main_content_text)

                        # Remove extra white space
                        main_content_text = ' '.join(
                            main_content_text.split())

                        # Replace HTML entities with their corresponding characters
                        main_content_text = html.unescape(
                            main_content_text)

                    else:
                        # Handle the case when 'main_content_list' is empty
                        main_content_text = "error"
                        err_log += f"Error in page structure, use body instead\n"
                        st.error(err_log)
                        sys.exit()
                except Exception as e:
                    err_log += f"Error while discivering page content\n"
                    return None

    # return as json
    return {
        "url": url,
        "text": main_content_text
    }, local_links

# Now you can work with the parsed content using Beautiful Soup


def main():
    global res, err_log, headers, sajt, source, vrsta, person_name, topic
    st.subheader('Pinecone Scraping')
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Set the domain URL
    col1, col2 = st.columns(2)
    with col1:
        with st.form(key='my_form', clear_on_submit=False):

            sajt = st.text_input("Unesi sajt : ")
            source = st.text_input("Unesi izvor : ")
            person_name = st.text_input("Unesi ime osobe : ")
            topic = st.text_input("Unesi temu : ")
            name_space = st.text_input("Unesi namespace : ")
            vrsta = st.radio("Unesi vrstu (default je body main): ",
                             ('body main', 'body'))
            submit_button = st.form_submit_button(label='Submit')
            if submit_button and not sajt == "" and not source == "" and not person_name == "" and not topic == "" and not name_space == "":
                res = requests.get(sajt, headers=headers)
                err_log = ""

                # Read OpenAI API key from file
                openai.api_key = os.environ.get('OPENAI_API_KEY')

                # Retrieving API keys from files
                PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

                # Setting the environment for Pinecone API
                PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

                pinecone.init(
                    api_key=PINECONE_API_KEY,
                    environment=PINECONE_API_ENV
                )

                # Initialize BeautifulSoup with the response text
                soup = BeautifulSoup(res.text, "html.parser")
                # soup = BeautifulSoup(res.text, 'html5lib')

                # Define a function to scrape a given URL

                links = [sajt]
                scraped = set()
                data = []
                i = 0
                placeholder = st.empty()
                ph2 = st.empty()
                with st.spinner(f"Scraping "):

                    while True:
                      # while i < 10:  # for test purposes
                        i += 1
                        if len(links) == 0:
                            st.success("URL list complete")
                            break
                        url = links[0]

                        # st.write(f'{url}, ">>", {i}')
                        placeholder.text(f'Obradjujem link broj {i}')
                        try:
                            res = scrape(url)
                            err_log += f" OK scraping {url}: {i}\n"
                        except Exception as e:
                            err_log += f"An error occurred while scraping {url}: page can not be scraped.\n"

                        scraped.add(url)

                        if res is not None:
                            page_content, local_links = res
                            data.append(page_content)
                            # add new links to links list
                            links.extend(local_links)
                            # remove duplicates
                            links = list(set(links))
                        # remove links already scraped
                        links = [
                            link for link in links if link not in scraped]

                    # tokenizer = tiktoken.get_encoding('p50k_base')

                    # create the length function

                    # Initialize RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=20,
                        length_function=tiktoken_len,
                        separators=["\n\n", "\n", " ", ""]
                    )

                chunks = []
                progress_text = "Embeding creation in progress. Please wait."
                progress_bar = st.progress(0.0, text=progress_text)
                ph = st.empty()
                # Iterate over data records
                for idx, record in enumerate(tqdm(data)):
                    # Split the text into chunks using the text splitter
                    texts = text_splitter.split_text(record['text'])
                    sto = len(data)
                    odsto = idx+1
                    procenat = odsto/sto
                    progress_bar.progress(
                        procenat, text=progress_text)
                    k = int(odsto/sto*100)
                    ph.text(
                        f'Ucitano {odsto} od {sto} linkova sto je {k} % ')
                    # Create a list of chunks for each text
                    chunks.extend([
                        {
                            'id': str(uuid4()),
                            'text': texts[i],
                            'chunk': i,
                            'url': record['url'],
                            'source': source,
                            'person_name': person_name,
                            'topic': topic
                        }
                        # Exclude chunks shorter than 200 characters
                        for i in range(len(texts)) if len(texts[i]) >= 200
                    ])

                # Set the embedding model name
                embed_model = "text-embedding-ada-002"

                # Set the index name and namespace
                index_name = 'embedings1'
                # Initialize the Pinecone index
                index = pinecone.Index(index_name)
                batch_size = 100  # how many embeddings we create and insert at once
                progress_text2 = "Upserting to Pinecone in progress. Please wait."
                progress_bar2 = st.progress(0.0, text=progress_text2)
                ph2 = st.empty()
                for i in tqdm(range(0, len(chunks), batch_size)):
                    # find end of batch
                    i_end = min(len(chunks), i+batch_size)
                    meta_batch = chunks[i:i_end]
                    # get ids
                    ids_batch = [x['id'] for x in meta_batch]
                    # get texts to encode

                    texts = [x['text'] for x in meta_batch]

                    # create embeddings (try-except added to avoid RateLimitError)
                    try:
                        res = openai.Embedding.create(
                            input=texts, engine=embed_model)

                    except:
                        done = False
                        while not done:
                            sleep(5)
                            try:
                                res = openai.Embedding.create(
                                    input=texts, engine=embed_model)
                                done = True

                            except:
                                pass

                    # cleanup metadata

                    cleaned_meta_batch = []  # To store records without [nan] embeddings
                    embeds = [record['embedding']
                              for record in res['data']]

                    # Check for [nan] embeddings
                    nan_indices = [index for index, emb in enumerate(
                        embeds) if np.isnan(emb).any()]
                    if nan_indices:
                        err_log += f"Records with [nan] embeddings:\n"
                        for index in nan_indices:
                            err_log += f"ID: {meta_batch[index]['id']}, Text: {meta_batch[index]['text']}\n"
                    # Filter out records with [nan] embeddings
                    cleaned_meta_batch.extend(
                        [meta_batch[i] for i in range(len(meta_batch)) if i not in nan_indices])

                    # Now cleaned_meta_batch contains records without [nan] embeddings

                    cleaned_meta_batch = [{
                        'text': x['text'],
                        'chunk': x['chunk'],
                        'url': x['url'],
                        'source': source,
                        'person_name': person_name,
                        'topic': topic
                    } for x in cleaned_meta_batch]

                    if embeds:
                        to_upsert = list(
                            zip(ids_batch, embeds, cleaned_meta_batch))
                    else:
                        err_log += f"Greska: {cleaned_meta_batch}\n"
                    # upsert to Pinecone
                    err_log += f"Upserting {len(to_upsert)} embeddings\n"
                    with open('err_log.txt', 'w', encoding='utf-8') as file:
                        file.write(err_log)
                    index.upsert(vectors=to_upsert,
                                 namespace=name_space)
                    stodva = len(chunks)
                    if i_end > i:
                        deo = i_end
                    else:
                        deo = i
                    progress = deo/stodva
                    l = int(deo/stodva*100)

                    ph2.text(
                        f'Ucitano je {deo} od {stodva} linkova sto je {l} %')

                    progress_bar2.progress(
                        progress, text=progress_text2)

                # gives stats about index
                with col2:

                    pinecone_stats(index)


if __name__ == "__main__":
    main()


