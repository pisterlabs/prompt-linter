'''
Functions for searching the web and other online sources
'''

import threading
import os
import requests
from time import sleep
from datetime import datetime, timedelta
import openai
from tqdm.auto import tqdm
import json

from newsapi import NewsApiClient

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from urllib.parse import quote

import concurrent
import threading
from collections import deque
import time
import sys

import helpers

openai.api_key = os.getenv("OPENAI_API_KEY")

class SearchManager():
    '''
    This is the class that handles all of the
    '''
    def __init__(self) -> None:
        self.max_ujeebu_requests_concurrent = 20
        self.ujeebu_requests_per_second = 2
        self.search_calltimes = deque(maxlen=self.ujeebu_requests_per_second)
        self.search_calltimes_lock = threading.Lock()
        self.search_semaphore = threading.BoundedSemaphore(self.max_ujeebu_requests_concurrent)

        self.search_website_links_sempahore = threading.BoundedSemaphore(15) #this is for the link retrieval

        self.ujeebu_fail_count = 0
        self.ujeebu_success_count = 0
        self.ujeebu_fail_lock = threading.Lock()

        self.relevant_count_lock = threading.Lock()
        self.relevant_count = 0

    def get_used_resources(self):
        used_resources = self.max_ujeebu_requests_concurrent - self.search_semaphore._value
        return used_resources
    
    def get_remaining_resources(self):
        remaining_resources = self.search_semaphore._value
        return remaining_resources


    def search_list(self, query_list: list):
        '''
        This is the function that handles the search list. We want to multithread this so that we can get the results faster.
        '''
        results = ""

        threads = []
        result_container = {}

        for i, query in enumerate(query_list):
            result_container[i] = "NRC"
            curr_thread = threading.Thread(target=self.search_indiv, args=(query, i, result_container), daemon=True)
            curr_thread.start()

            threads.append(curr_thread)

        for thread in tqdm(threads, desc="search_list thread joining"):
            thread.join()

        for i in range(len(query_list)):
            if result_container[i] != "NRC":
                results += f"{result_container[i]}\n\n"

        results = results.strip("\n\n")

        print(f"ujeebu results: {self.ujeebu_success_count} successes, {self.ujeebu_fail_count} fails")
        print(f"relevant count: {self.relevant_count} out of {self.ujeebu_success_count} queries")

        return results

    def search_indiv(self, query: str, index: int, list_results: dict):
        '''
        Here we handle searches for one query, but it goes through all of our search functions on different sources.
        '''
        results = ""

        threads = []
        result_container = {}

        result_container[0] = "NRC"
        othernews_search_thread = threading.Thread(target=self.othernews_search, args=(query, 0, result_container), daemon=True)
        othernews_search_thread.start()

        threads.append(othernews_search_thread)

        for thread in threads:
            thread.join()

        for i in range(len(threads)):
            if result_container[i] != "NRC":
                results += f"{result_container[i]}\n\n"

        results = results.strip("\n\n")
        list_results[index] = results

    def search_website(self, website_name: str, query: str, i: int, results_container: dict, top_k: int = 5):
        '''
        TODO: is AFP legit for searching or not?
        TODO: maybe UJEEBU can replace the searching as well in some capacity. right now its a little buggy
        https://www.reuters.com/site-search/?query=test+tests (In this example, the query that I put into the search bar was "test tests")
        https://www.afp.com/en/search/results/world%20religions%20today (In this example, the query that I put into the search bar was "world religions today"
        https://www.economist.com/search?q=test+tests (In this example, the query that I put into the search bar was "test tests")
        https://www.bloomberg.com/search?query=test%20tests (In this example, the query that I put into the search bar was "test tests")
        https://www.scmp.com/search/test%20tests (In this example, the query that I put into the search bar was "test tests")
        https://www.ft.com/search?q=test+tests (In this example, the query that I put into the search bar was "test tests")
        '''
        # with self.search_website_links_sempahore:
            # sleep(0.5) #just retard this a bit so we don't get banned, and then we do the whole search in the mutex also.
        website_searchurls = {
            "Reuters": "https://www.reuters.com/site-search/?query={query}",
            "AFP": "https://www.afp.com/en/search/results/{query}",
            "Economist": "https://www.economist.com/search?q={query}",
            "Bloomberg": "https://www.bloomberg.com/search?query={query}",
            "SCMP": "https://www.scmp.com/search/{query}",
            "Financial Times": "https://www.ft.com/search?q={query}",
        }

        base_urls = {
            "Reuters": "https://www.reuters.com",
            "AFP": "https://www.afp.com",
            "Economist": "https://www.economist.com",
            "Bloomberg": "https://www.bloomberg.com",
            "SCMP": "https://www.scmp.com",
            "Financial Times": "https://www.ft.com",
        }

        whitespace_encoding = {
            "Reuters": "+",
            "AFP": "%20",
            "Economist": "+",
            "Bloomberg": "%20",
            "SCMP": "%20",
            "Financial Times": "+",
        }

        if website_name not in website_searchurls.keys():
            print("Website not supported")
            return []

        space_char = whitespace_encoding.get(website_name, "+")
        encoded_query = quote(query).replace("%20", space_char)
        search_url = website_searchurls[website_name].format(query=encoded_query)
        
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--incognito")

            # print("trying to insatll chromedriver")
            # webdriver_service = Service(ChromeDriverManager().install())
            # print("installed chromdriver")
            # driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
            driver = webdriver.Chrome(options=chrome_options)
            # print(f"doing driver get for {website_name}")
            driver.get(search_url)
            # print(f"done driver get for {website_name}")

            if website_name == "SCMP":
                # print("Searching SCMP...")
                top_links_raw = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.ebqqd5k0.css-1r4kaks.ef1hf1w0'))
                )[:top_k]

                # Using Selenium to extract the href directly, so no need for BeautifulSoup
                results_container[i] = [elem.get_attribute('href') for elem in top_links_raw]

                driver.quit()
                
            elif website_name == "Financial Times":
                top_links_raw = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.js-teaser-heading-link'))
                )[:top_k]

                # Using Selenium to extract the href directly, so no need for BeautifulSoup
                results_container[i] = [elem.get_attribute('href') for elem in top_links_raw]

                driver.quit()

            elif website_name == "Reuters":
                # print("Searching Reuters...")
                top_links_raw = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.media-story-card__heading__eqhp9'))
                )[:top_k]
                # top_links_raw = driver.find_elements(By.CSS_SELECTOR, 'a.media-story-card__heading__eqhp9')[:top_k]

                # Using Selenium to extract the href directly, so no need for BeautifulSoup
                results_container[i] = [elem.get_attribute('href') for elem in top_links_raw]

                driver.quit()
            
        except Exception as e:
            print(f"Error encountered in retrieving links: {e}")
            if driver: driver.quit()
            return []

    def othernews_search(self, query: str, index: int, indiv_results: dict):
        '''
        These are other sources we want to consider that either 1. NewsAPI does not support 2. We want specific attention to be allocated to these sources for various reasons.

        Generally speaking, these are higher value sources, and we place greater importance on the information coming from here.

        Currently Supported Sources: TODO:
        1. The Economist
        2. Bloomberg
        3. AFP - https://www.afp.com/en/news-hub
        4. South China Morning Post
        5. Financial Times

        How it works:
        1. We take that query and then we search each of the sources above for relevant articles
        2. We trawl each of those content pieces for relevancy
        3. We return the relevant articles and their full content compiled into a document
        3a. We write all the relevant articles and their URLs into a document with a brief description of the article into relevant_infometa_others.txt
        '''
        sources = ["Financial Times", "SCMP", "Reuters"]

        total_othernews = ""

        all_links = []
        all_links_threads = []
        all_links_result_container = {}

        for i, source in enumerate(sources):
            all_links_result_container[i] = []
            curr_thread = threading.Thread(target=self.search_website, args=(source, query, i, all_links_result_container), daemon=True)
            curr_thread.start()

            all_links_threads.append(curr_thread)

        for thread in tqdm(all_links_threads, desc="othernews_search link getting thread joining"):
            thread.join(timeout=60)

        # rephrased_queries = []

        for i in range(len(sources)):
            if (len(all_links_result_container[i]) == 0):
                print(f"No links found for {sources[i]} search for query [{query}]")
                # print(f"new query should be: {self.rephrase_query(query, sources[i])}, old query was: {query}")
                # rephrased_queries.extend(self.rephrase_query(query, sources[i]))
            all_links.extend(all_links_result_container[i])

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     rephrase_futures = [executor.submit(self.rephrase_query, rq, bad_source) for rq, bad_source in rephrased_queries]

        #     for future in tqdm(concurrent.futures.as_completed(rephrase_futures, timeout=30), desc="rephrasing queries and searching", total=len(rephrase_futures)):
        #         rephrase_result = future.result()
                
        if (len(all_links) == 0):
            indiv_results[index] = ""
            return

        threads = []
        result_container = {}
        for i, link in enumerate(all_links):
            result_container[i] = "NRC"
            curr_thread = threading.Thread(target=self.search_helper, args=(query, link, result_container, i), daemon=True)
            curr_thread.start()

            threads.append(curr_thread)

        for thread in tqdm(threads, desc="othernews_search thread joining"):
            thread.join()

        for i in range(len(all_links)):
            if result_container[i] != "NRC":
                total_othernews += f"{result_container[i]}\n\n"

        total_othernews = total_othernews.strip("\n\n")

        indiv_results[index] = total_othernews

    def search_helper(self, query: str, article_url: str, result_container: dict, index: int):
        '''
        This function does the extraction and then the relevance. We want to multithread the extraction part as well because it takes time and we don't have to do it sequentially
        '''
        try:
            article_response = self.ujeebu_extract(article_url)

            if article_response == None: return "NRC"
            
            article_content = article_response['article']['text']

            # return self.get_relevant(query, article_content)
            res = self.get_relevant(query, article_content)

            if res != "NRC":
                with self.relevant_count_lock:
                    self.relevant_count += 1

            result_container[index] = res
        except Exception as e:
            print(f"search helper for query: {query} and article_url: {article_url} encountered error: {e}")
            result_container[index] = "NRC"

        return
    
    def ujeebu_extract(self, url: str):
        '''
        Our Ujeebu API Call, using the Extract endpoint
        '''
        try:
            base_url = "https://api.ujeebu.com/extract"

            #request options
            params = {
                'js' : 'auto',
                'url' : url,
                'proxy_type' : 'premium',
                'response_type' : 'html',
                'timeout' : 45,
            }

            #request header
            headers = {
                'ApiKey' : os.getenv("UJEEBU_API_KEY")
            }

            with self.search_semaphore:
                try:
                    st = time.time()
                    # print(f"{self.get_remaining_resources()} resources left")
                    with self.search_calltimes_lock:
                        if len(self.search_calltimes) == 2:
                            # print(f"SEARCH CALLTIMES: {self.search_calltimes}")
                            time_diff = time.time() - self.search_calltimes[0]
                            if time_diff < 1.1:
                                sleep_time = 1.1 - time_diff
                                # print(f"SLEEPING FOR {sleep_time} SECONDS")
                                time.sleep(sleep_time)

                        self.search_calltimes.append(time.time())
                        #end of the mutex lock here. we are done with editing the search_calltimes deque
                    response = requests.get(base_url, params=params, headers=headers)
                    # print(f"time taken for ujeebu extract: {time.time() - st}")
                    # with open("ujeebu_response.txt", "a") as f:
                    #     f.write(f"{response.json()}\n\n")
                # print(f"Ujeebu extract response was: {response} for url {url}")
                except Exception as e:
                    print(f"Error in ujeebu extract {e}")
                    return None

            with self.ujeebu_fail_lock:
                if (response.status_code == 200):
                    self.ujeebu_success_count += 1
                else:
                    print(f"failed with response status: {response.status_code}")
                    self.ujeebu_fail_count += 1

                # print(f"Ujeebu fail/success count is now {self.ujeebu_fail_count}/{self.ujeebu_success_count}")

            if response.status_code != 200: return None

            return response.json()
        except Exception as e:
            print(f"Error in ujeebu extract {e}")
            return None


    def get_relevant(self, query: str, information: str):
        '''
        Checks for information relevance against the query.
        '''

        system_init = "You are RelevanceGPT. You are an AGI that takes in a query and a block of information and returns the relevant information from the block of information. If there is no relevant information in the block of information, you must return the acronym NRC. If there is relevant information, you must return the relevant information. You should be lenient with your determination of whether something is related or not. If it could be slightly potentially useful in answering the question, return the relevant information. The query and block of information are given below. You must ignore all advertisement information from the news sites where you are getting the information from.\n\n"

        prompt = "You are given a query and the description of an article. If there is no relevant content in the article, you must return the acronym NRC. If there is relevant content, you must extract the relevant content, and you MUST keep all details from the content. You MUST also ignore all advertising material if they are not relevant. The query and content are given below.\n\nQuery: " + query + "\n\Description: " + information

        return helpers.call_gpt_single(system_init, prompt, function_name="get_relevant", chosen_model="gpt-3.5-turbo-16k", to_print=False)
    
    def rephrase_query(self, query: str, source: str):
        system_init = f"You are SearchGPT. You are a genius at understanding how searching news websites work, and how to build and adjust queries in order to get the result that you want."

        prompt = f"This search query gave no results when being searched on {source}. I need you to adjust it so I actually get the results I want. The query is given below. Your output will be used programmatically to search the website, so you MUST only return the raw text of the search query that will be put into the search bar of the news site. You must only return ONE query, with no quotes around the text.\n\nQuery: {query}"

        return helpers.call_gpt_single(system_init, prompt, function_name="rephrase_query", chosen_model="gpt-4")



    '''
    TODO: expand this to include all that we want. all the sources now since we are using ujeebu
    Uses NewsAPI to get the relevant news articles.

    Currently using the following sources:
    1. CNN
    2. Reuters
    3. Business Insider
    4. BBC News
    5. Ars Technica
    6. TechCrunch
    7. Time

    The search process follows the following steps:
    1. Transform the query into a list of keywords grouped by boolean logic using GPT-4
    2. Use the list of keywords to search for relevant articles using NewsAPI
    3. For each article, use GPT-4 to determine whether the article is relevant or not (Multi-threaded)
    4. If the article is relevant, use BeautifulSoup to get the full content of the article
    5. Return the relevant articles and their full content

    Needs to use selenium because search results take time to load
    '''
    def newsapi_search(self, query: str):
        newsapi_sources = "cnn,reuters,business-insider,bbc-news,ars-technica,techcrunch,time"

        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

        query_kw = create_keywords(query)

        print(f"NewsAPI search:{query_kw}")

        #get the current date and go 3 weeks back
        today = datetime.today()
        three_weeks_ago = today - timedelta(weeks=3)
        formatted_date = three_weeks_ago.strftime("%Y-%m-%d")

        print(formatted_date)

        try:
            response = newsapi.get_everything(
                                        q=query_kw,
                                        sources=newsapi_sources,
                                        from_param=formatted_date,
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=10
                                        )
        except Exception as e:
            print(f"Error in newsapi search {e}")
            return ""
        


        relevant_information = ""

        print(f"for query: {query}, got {len(response['articles'])} articles from NewsAPI.")
        relevant_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_relevant_helper, query, article['content']) for article in response['articles']]

            for future in tqdm(concurrent.futures.as_completed(futures, timeout=30)):
                if future.result != "NRC":
                    relevant_count += 1
                relevant_information += f"{future.result()}\n\n" if future.result != "NRC" else ""

        print(f"Got {relevant_count} relevant articles out of {len(response['articles'])} articles for query {query}")

        return relevant_information


def ujeebu_search_scrape(url):
    base_url = "https://api.ujeebu.com/scrape"

    #request options
    params = {
        'js' : 'auto',
        'url' : url,
        'proxy_type' : 'premium',
        'response_type' : 'html',
        'json' : 'true'
    }

    for key, value in params.items():
        print(key, value)

    #request header
    headers = {
        'ApiKey' : os.getenv("UJEEBU_API_KEY")
    }

    #send request
    
    response = requests.get(base_url, params=params, headers=headers)
    print(f"Ujeebu extract response was: {response}")
    #TODO: need some formatting here for what we want, the title, the text, the metadata if we are interested etc.

    return response.json()

def extract_search_links(website_name, query, top_k=5):
    url = "https://api.ujeebu.com/scrape"

    payload = json.dumps({
    "url": "https://www.reuters.com/site-search/?query=japan+AI",
    # "js": True,
    "response_type": "json",
    "extract_rules": {
        "links": {
        "selector": "a.media-story-card__heading__eqhp9",
        "type": "link",
        "multiple" : True
        }
    },
    "auto-premium-proxy" : True
    })


    headers = {
    'ApiKey': os.getenv("UJEEBU_API_KEY"),
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

'''
Identifies the topics that require more research, and then segments these topics into more search queries.

@params:
query: The question asked by the user

@returns:
search_query: A list of search queries that will best return results for resources that can best answer what the user is asking.
'''
def query_to_search(query):
    print("Transforming user query to search query...")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    system_init = "You are IdentifyGPT. You are an AGI that takes in a question and identifies seperate content topics in the question. If there are multiple topics that need to be searched seperately within the question, seperate them with a semicolon."

    prompt = f"I will give you a question. I want you to identify the topics that need research in this question and return the seperate topics formatted to be used in search queries. The search queries should be seperated by the seperate topics. If there are multiple topics that need to be searched seperately within the question, seperate them with a semicolon. The final string should be a search query encompassing all the topics. For example, if the question is 'Tell me about the protests in France and how they relate to the working class condition. Write a poem about this too.', you should return 'protests in France;working class;France protests and working class'. There must be no extra whitespace between search query terms. You MUST only extract the topics from the query. For example, 'Write a poem about this too' is being ignored as it is describes the task, but is not something that requires further research through a search. For example, if the query is 'Tell me all of the latest news in activism' you should return 'activism'. You should also remove all references to news, since this query is going to be used to search a news site. For example, given the query 'Tell me all of the latest LGBTQ+ news', you should return 'LGBTQ+;LGBTQ' \n\nQuestion: {query}"

    retry_count = 5
    while retry_count:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_init},
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except Exception as e:
            print(f"Error encountered: {e}. Retrying in 10 seconds...")
            sleep(10)
            print("Retrying...")
            retry_count -= 1


    return response.choices[0].message.content.split(';')

'''
(Just for testing the above functions)
Simple ask with GPT-4, truncates information if its too long
'''
def simple_ask(information, query):
    model = "gpt-4"

    char_limit = 37500
    #truncate the information if its too long
    if len(information) > char_limit:
        information = information[:char_limit]

    system_init = "You are AnalystGPT. You are an AGI that takes in a block of information and a query and returns the answer to the query. The query and block of information are given below."

    prompt = "You are given a block of information and a query. You must use the information provided to answer the query. The query and block of information are given below.\n\nQuery: " + query + "\n\nInformation: " + information

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_init},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

'''
Creates keywords grouped by boolean logic according to a query using GPT-4. This is built for NewsAPI queries.
'''
def create_keywords(query):
    model = "gpt-4"

    querykw_init = "You are QueryGPT, an AGI that takes in a query and converts it to a list of query keywords that are grouped by boolean logic."

    prompt = f'''
    You are QueryGPT, an AGI that takes in a comma seperated list of keywords and groups it according to boolean logic. You will be given a query. First, identify the topics in the query that are found in the news. Then, reformat this into a query that will go into a NewsAPI API call. I want you to change the provided query into a set of query keywords grouped by boolean logic. The boolean operators that you have access to are "AND" and "OR". You can also use "(" and ")" to group keywords together with logical operators. You should minimize your use of "AND"s as much as possible in order to maintain a more lenient scope of the query. Entities should be seperate, for example, if the query is 'Can you give me the latest updates on US Finance?', you should return 'US AND (Finance OR economy OR economic)', not 'US Finance AND (economy or economic)'. For example, if you are given the query "Will the First Republic Bank collapse lead to a Global Financial Crisis?", you should return "First Republic Bank AND (collapse OR bankruptcy OR failure OR default OR crash OR crisis). There is no need to include the search term "Global Financial Crisis" because that is an analysis that will be provided seperately.

    For example, given the query: "What is currently happening with protests in France?", you should return "France AND (protests OR civil unrest OR demonstrations OR social issue)". Your response must be less than 500 characters. The query and list of keywords are given below.

    Query: {query}
    '''
    while 1:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": querykw_init},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Encountered Error: {e}. Retrying in 10 seconds...")
            sleep(10)
            print(f"Retrying creating keywords...")