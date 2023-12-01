import requests
import os
import json
import aiohttp
import asyncio
import pdfplumber
import io
import logging
import tiktoken
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def extract_html(google_json):
    # extract important text from google_json
    try:
        extracted_html = "\n Extracted HTML: \n"
        for url, text in google_json.items():
            if text is not None: # We only want URLs with text
                extracted_html += f"URL: {url}\n"
                extracted_html += f"Text: {text}\n"
    except Exception as e:
        print(f"Error in extracting HTML: {e}")

    return extracted_html

def count_tokens(extracted_html):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_count = len(encoding.encode(extracted_html))
    return token_count

def analyze_google_json(user_question, snippets, google_json):
    system_prompt = f"The user just searched google for '{user_question}'You are going to answer the user's question using the text extracted from the following google results. Think step by step when you answer. Return the links you used in your analysis. Return your answer in well formatted Markdown."

    extracted_html = extract_html(google_json)
    
    anthropic = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    completion = anthropic.completions.create(
    model="claude-instant-1.1-100k",
    max_tokens_to_sample=100000,
    prompt=f"{HUMAN_PROMPT} {system_prompt} \n Google Snippets {snippets} \n Extracted Link Text: {extracted_html} {AI_PROMPT}",
    )

    response = completion.completion

    return response

def search_google(user_question):

    params = {
        "q": f"{user_question}",
        "hl": "en",
        "gl": "us",
        "num": "6",
        "google_domain": "google.com",
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    
    search_results_as_str = ""
    if 'answer_box' in results:
        answer_box = results['answer_box']
        title = answer_box.get('title', 'No title available')
        url = answer_box.get('link', 'No URL available')
        snippet = answer_box.get('snippet', 'No snippet available')
        search_results_as_str += f"\nAnswer Box:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"

    for i, result in enumerate(results['organic_results'], start=1):
        title = result.get('title', 'No title available')
        url = result.get('link', 'No URL available')
        snippet = result.get('snippet', 'No snippet available')
        search_results_as_str += f"\nResult {i}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"

    snippets = search_results_as_str

    search_results = []

    if 'answer_box' in results:
        search_results.append(results['answer_box'])

    for result in results['organic_results']:
        search_results.append(result)

    return search_results, snippets

async def fetch_url_async(session, url):
    try:
        logging.info(f"Attempting to fetch URL {url}")
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                # Get content type
                content_type = response.headers.get('content-type', '').lower()
                if 'html' in content_type:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    extracted_text = ' '.join([tag.get_text() for tag in texts])
                    logging.info(f"Successfully fetched and parsed HTML from {url}")
                    return (url, extracted_text)
                elif 'pdf' in content_type:
                    # Extract text from PDF
                    pdf_data = await response.read()
                    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                        extracted_text = ' '.join(page.extract_text() for page in pdf.pages)
                        logging.info(f"Successfully fetched and parsed PDF from {url}")
                        return (url, extracted_text)
                else:
                    logging.error(f"Unsupported content type {content_type} for URL {url}")
                    return (url, None)
            else:
                logging.error(f"Failed to fetch URL {url}. Status code: {response.status}")
                return (url, None)
    except Exception as e:
        logging.error(f"Failed to fetch URL {url}. Error: {e}")
        return (url, None)


async def store_html_as_text(search_results):
    urls = [result.get('link', '') for result in search_results if 'link' in result]
    print(f"Found {len(urls)} URLs")

    google_json = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_url_async(session, url))

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                google_json[result[0]] = result[1]
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))

    return google_json

def search_main(google_query):
    # Perform the Google search
    search_results, snippets = search_google(google_query) # search results = list of dictionaries

    # Create a new asyncio loop instance
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    google_json = loop.run_until_complete(store_html_as_text(search_results)) # google_json = dictionary of url: text

    extracted_html = extract_html(google_json)

    token_count = count_tokens(extracted_html)

    print(f"Token count for extracted URL HTML text: {token_count}")

    if token_count > 90000:
        print("Token count is too high. Please try again with a different query. TO DO - FIX LATER. Add alternative (maybe summarize in chunks or do neighbor comparison and only summarize chunks) ")
        return True
    
    else:
        analyzed_google_results = analyze_google_json(google_query, snippets, google_json)

        analyzed_google_results += "\n\n Above is the analyzed search results from google which help anwswer the user's question."

        return analyzed_google_results, snippets