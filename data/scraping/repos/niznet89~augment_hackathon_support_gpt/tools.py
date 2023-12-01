import deeplake
from llama_index.readers.deeplake import DeepLakeReader
from llama_index import VectorStoreIndex, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
import random
from dotenv import load_dotenv
import os
import openai
import requests
import json
from bs4 import BeautifulSoup
from llama_index import Document
import cohere


load_dotenv()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
activeloop_key = os.environ.get("ACTIVELOOP_TOKEN")
scraping_dog_key = os.environ.get("SCRAPING_DOG_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
zendesk_api = os.environ.get("ZENDESK_API")
zendesk_email = os.environ.get("ZENDESK_EMAIL")


co = cohere.Client(cohere_api_key)


os.environ["OPENAI_API"] = openai_api_key
os.environ[
    "ACTIVELOOP_TOKEN"
] = activeloop_key

openai.api_key = openai_api_key


def search_discord(query):
    """Useful to search in Discord and see if thet question has been asked and answered already by the community"""

    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    reader = DeepLakeReader()
    query_vector = [random.random() for _ in range(1536)]
    documents = reader.load_data(
        query_vector=query_vector,
        dataset_path="hub://tali/ocean_protocol_discord",
        limit=30,
    )
    documents = documents

    dict_array = []
    for d in documents:
        insert = {"text": d.text}
        dict_array.append(insert)

    response = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=dict_array,
        top_n=3,
    )

    document_array = []

    for doc in response:

        url_prompt = f"""You are an expert at parsing out information from data based on a query. Here is a data source: {cut_string_at_char(doc.document['text'])}

                        Here is the query: {query}

                        ONLY return text that is relevant to answering the query.
                        DO NOT alter the text in any capacity, only return it as it is presented."""

        chat_message = {"role": "user", "content": url_prompt}
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[chat_message],
            temperature=0
        )

        completion_string = completion.choices[0].message['content']
        print(completion_string)
        document = Document(text=completion_string,
                            extra_info={'source': "test.com"})
        document_array.append(document)

    return document_array


def google_search(query):
    """Useful if you want to search the Web - you will need to enter an appropriate search query to get more information"""

    num_results = 6

    google_url = f'https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num={num_results}'

    google_response = requests.get(google_url)

    google_data = json.loads(google_response.text)

    prompt = f"""Your job is to select the UP TO the 3 most relevant URLS related to this query based on the available context provided: {query}.

                Here is the data to parse the URLS out of: {str(google_data)}

                ONLY return the 1-3 URLS, with each one seperated by a comma.

                ONLY return the URL if it looks like it is relevant to the query or would be helpful in answering the query.

                Example: https://example1.com,https://example2.com,https://example3.com"""
    chat_message = {"role": "user", "content": prompt}

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[chat_message],
        temperature=0
    )
    completion_string = completion.choices[0].message['content']
    print(completion_string)
    completion_array = completion_string.split(",")

    document_array = []

    for url in completion_array:
        payload = {'api_key': scraping_dog_key, 'url': url, 'dynamic': 'true'}
        resp = requests.get(
            'https://api.scrapingdog.com/scrape', params=payload)

        url_prompt = f"""You are an expert at parsing out information from data based on a query. Here is a scraped URL: {cut_string_at_char(resp.text)}

                        Here is the query: {query}

                        ONLY return text that is relevant to answering the query.
                        DO NOT alter the text in any capacity, only return it as it is presented."""

        chat_message = {"role": "user", "content": url_prompt}
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[chat_message],
            temperature=0
        )
        completion_string = completion.choices[0].message['content']
        print(completion_string)
        document = Document(text=completion_string, extra_info={'source': url})
        document_array.append(document)
    print(document_array)
    return document_array


def ticket_escalation(email, query):
    """Use this Tool (ticket escalation) if you cannont answer the question. Do not continue with any further iterations. If this tool is used, end with: 'Query Escalated'"""

    prompt = f"You are an expert at writing ticket Subject lines. Based on the question, write a brief 1 line summary that fits in a subject line. {query}"
    chat_message = {"role": "user", "content": prompt}
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[chat_message],
        temperature=0
    )

    completion_string = completion.choices[0].message['content']
    # New ticket info
    subject = f'AI BOT ESCALATION: {completion_string}'
    body = f"USER EMAIL: {email}\n\n" + query

    # Package the data in a dictionary matching the expected JSON
    data = {'ticket': {'subject': subject, 'comment': {'body': body}}}

    # Encode the data to create a JSON payload
    payload = json.dumps(data)

    # Set the request parameters
    url = 'https://taliaihelp.zendesk.com/api/v2/tickets.json'
    user = zendesk_email
    pwd = zendesk_api
    headers = {'content-type': 'application/json'}

    # Do the HTTP post request
    response = requests.post(
        url, data=payload, auth=(user, pwd), headers=headers)

    # Check for HTTP codes other than 201 (Created)
    if response.status_code != 201:
        print('Status:', response, 'Problem with the request. Exiting.')
        exit()

    # Report success
    print('Successfully created the ticket.')


def ticket_solved(email, query):
    """Use this Tool (Ticket Solved) if you CAN answer the question. Do not continue with any further iterations. If this tool is used, end with: 'Query Escalated'"""

    prompt = f"You are an expert at writing ticket Subject lines. Based on the question, write a brief 1 line summary that fits in a subject line. {query}"
    chat_message = {"role": "user", "content": prompt}
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[chat_message],
        temperature=0
    )

    completion_string = completion.choices[0].message['content']
    # New ticket info
    subject = f'TICKET SOLVED: {completion_string}'
    body = f"USER EMAIL: {email}\n\n" + query

    # Package the data in a dictionary matching the expected JSON
    data = {'ticket': {'subject': subject, 'comment': {'body': body}}}

    # Encode the data to create a JSON payload
    payload = json.dumps(data)

    # Set the request parameters
    url = 'https://taliaihelp.zendesk.com/api/v2/tickets.json'
    user = zendesk_email
    pwd = zendesk_api
    headers = {'content-type': 'application/json'}

    # Do the HTTP post request
    response = requests.post(
        url, data=payload, auth=(user, pwd), headers=headers)

    # Check for HTTP codes other than 201 (Created)
    if response.status_code != 201:
        print('Status:', response, 'Problem with the request. Exiting.')
        exit()

    # Report success
    print('Successfully created the ticket.')


def cut_string_at_char(input_string, max_tokens=14000):
    length = len(input_string)

    if length > max_tokens:
        tokens = input_string[:max_tokens]
        return tokens
    else:
        return input_string


__all__ = ['search_discord', 'google_search',
           'ticket_escalation', "ticket_solved"]
