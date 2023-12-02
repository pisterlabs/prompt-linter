import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain


# tried install lib per thread below, couldn't find python-magic-bin, so installed python-magic instead
# https://stackoverflow.com/questions/76247540/loading-data-using-unstructuredurlloader-of-langchain-halts-with-tp-num-c-bufpip 

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import yfinance as yf
import requests
from stqdm import stqdm

import time
from datetime import datetime

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import config as config

GPT_3_5_TOKEN_LIMIT = 4096

def display_news_analysis(portfolio_summary):
    input_container = st.container()
    output_container = st.container()

    
    with input_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.form(key='search_form'):
                if not config.check_for_api_key('openai'):
                    label = "Enter [OpenAI API Key](https://platform.openai.com/account/api-keys) to interpret news"
                    temp_key = st.text_input(label, value=config.get_api_key('openai'), type="password")
                    if temp_key:
                        config.set_api_key('openai', temp_key)
                        
                # TODO: add Bing, Google to this list - and perhaps a multi-select?
                st.write("Alpha Vantage is free, targeted financial news, Serper is paid, open internet search (todo: add Bing, Google, etc.)")
                st.radio("News Source", ['Serper', 'Alpha Vantage'], index=0, key='news_source')
                
                if st.session_state.news_source == 'Alpha Vantage':
                    if not config.check_for_api_key('alpha_vantage'):
                        label = "Enter [Alpha Vantage API Key](https://www.alphavantage.co/) to retrieve news and sentiment from the internet (free service)"
                        temp_key = st.text_input(label, value=config.get_api_key('alpha_vantage'), type="password")
                        if temp_key:
                            config.set_api_key('alpha_vantage', temp_key)
                else:
                    if not config.check_for_api_key('serper'):
                        label = "Enter [Serper API Key](https://serper.dev/api-key) to retrieve news from the internet (paid service)"
                        temp_key = st.text_input(label, value=config.get_api_key('serper'), type="password")
                        if temp_key:
                            config.set_api_key('serper', temp_key)
                        
                        
                st.write("Leverage OpenAI to interpret news and summarize it, currently experiencing some hangs pulling news from the web, so be aware")   
                search_and_summarize_w_openai = st.form_submit_button("Search & Summarize with OpenAI")
            st.caption("*Search & Summarize: Uses Langchain, OpenAI APIs and Serper or Alpha Vantage, to search the web for news and summarize search results.*")
        

    with output_container:
        st.markdown("""---""")
        n_tickers = len(portfolio_summary['tickers'])
            
        col1, col2, col3 = st.columns(3)
        column_map = {0: col1, 1: col2, 2: col3} 
                
        if config.check_for_api_key('openai') and (config.check_for_api_key('serper') or config.check_for_api_key('alpha_vantage')):
            if search_and_summarize_w_openai:
                progress_bar = st.progress(0)
                completed_tickers = 0
                column_counter = 0
                n_cores = multiprocessing.cpu_count()

                with st.spinner("Searching the web for news and summarizing it with OpenAI's GPT-3..."):
                    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_cores, n_tickers)) as executor:
                        futures_to_indices = {}
                        futures = []
                        for ticker_index, ticker in enumerate(portfolio_summary['tickers']):
                            future = executor.submit(get_news_for_ticker_and_analyze, ticker, st.session_state.news_source)
                            futures.append(future)
                            futures_to_indices[future] = ticker_index  # remember the ticker index along with the future

                        for future in concurrent.futures.as_completed(futures):
                            ticker_index = futures_to_indices[future]
                            news_for_ticker, news_results = future.result()
                            logger.debug(f"Process returned news for: {news_for_ticker}")
                            st.session_state[f'{news_for_ticker}_news_and_analysis'] = news_results

                            # As soon as we get the news for a ticker, display it in the appropriate column.
                            if news_results and len(news_results) > 0:  # Only increment column_counter for non-empty news results
                                with column_map[column_counter % 3]:
                                    logger.debug(f"Displaying news as it arrives for {ticker}")
                                    display_news_for_ticker(news_for_ticker)
                                column_counter += 1
                            
                            completed_tickers += 1
                            progress_bar.progress(completed_tickers / n_tickers)
            else:
                logger.debug("Search & Summarize not clicked, painting what's in session state")
                # form wasn't submitted, but we may have news to display
                column_counter = 0
                for ticker_index, ticker in enumerate(portfolio_summary['tickers']):
                    if f'{ticker}_news_and_analysis' in st.session_state:
                        news_results = st.session_state[f'{ticker}_news_and_analysis']
                        if news_results and len(news_results) > 0:
                            with column_map[column_counter % 3]:
                                display_news_for_ticker(ticker)
                            column_counter += 1
        else:
            st.error("Please enter API keys for OpenAI and Serper or Alpha Vantage to search and summarize news")

# result limit based on rate limits of the LLM model - if run locally, can increase this
def get_news_for_ticker_and_analyze(ticker, news_source, n_search_results=3):
    news_results = []
    try:
        logger.debug(f"Getting news for {ticker}")
        #company = yf.Ticker(ticker).info['longName']
        company = get_company_name(ticker)
        
        # prioritize news from alpha vantage as it includes sentiment - TODO: add sentiment score processing & a toggle for news source?
        if news_source == 'Alpha Vantage':
            logger.debug(f"Getting news from Alpha Vantage for {ticker}")
            result_dict = get_news_and_sentiment_from_alpha_vantage(ticker, n_search_results)
        else:
            logger.debug(f"Getting news from Serper for {ticker}")
            result_dict = get_news_from_serper(ticker, company, n_search_results)

        if not result_dict or not result_dict['news']:
            logger.error(f"No search results for: {ticker}.")
        else:
            # Load URL data from the news search
            for i, item in zip(range(n_search_results), result_dict['news']):
                try:
                    logger.info(f'{ticker}: processing news item {i} for company {company} from link {item["link"]} with keys: {item.keys()}')
                    # TODO: appears to hang sometimes... particularly with the open web results vs. the Vantage results
                    loader = UnstructuredURLLoader(urls=[item['link']], continue_on_failure=False)
                    data = loader.load()
                    logger.debug(f'{ticker}: done processing news item {i} for company {company} from link {item["link"]}')
                                   
                    summary = "No summary available"     
                    # Truncate the data to 4096 characters
                    if isinstance(data, list):
                        for i, element in enumerate(data):
                            # If the element is a Document object, extract and truncate the text
                            #logger.debug(f"Element {i} is type: {type(element)}")
                            if isinstance(element, Document):
                                #logger.debug(f"Element {i} is a Document object\n{element}")
                                element.page_content = element.page_content[:GPT_3_5_TOKEN_LIMIT]
                                #logger.debug(f"Truncated data: {data}")
                                break
                            else:
                                logger.debug(f"Element {i} is not a Document object\n{element}")
            
                        # to help with rate limiting
                        time.sleep(1)
                        
                        logger.info(f"Calling OpenAI to summarize news about {company} w/ticker {ticker}")
                        # Initialize the ChatOpenAI module, load and run the summarize chain
                        llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=config.get_api_key('openai'))
                        chain = load_summarize_chain(llm, chain_type="map_reduce")
                        ai_summary = chain.run(data)
                        ai_summary = ai_summary.replace("$", "\\$")

                    # TODO: likely augment the return with dates if they're available and potentially other info, eg, sentiment
                    # if from a news source that provides summaries, eg. Alpha Vantage, use that too

                        
                except Exception as e:
                    ai_summary = f'Error while summarizing ({e})'
                    logger.error(f"Exception summarizing news about {company} w/ticker {ticker}: {e}")
                    
                if 'summary' in item:
                    news_results.append({'title': item['title'], 'link': item['link'], 'date': item['date'], 'summary': item['summary'], 'ai_summary': ai_summary})
                else:
                    news_results.append({'title': item['title'], 'link': item['link'], 'date': item['date'], 'ai_summary': ai_summary})
                    
    except Exception as e:
        logger.error(f"Exception searching for news about {company} w/ticker {ticker}: {e}")

    logger.debug(f"Completed getting news for {ticker}")
        
    return (ticker, news_results)

def get_news_from_serper(ticker, company, n_search_restuls=3):
    result_dict = {}
    try:
        search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=config.get_api_key('serper'))
        search_query = f"financial news about {company} or {ticker}"
        logger.debug(f"Search query: {search_query}")
        
        # search hangs sometimes... trying sleep
        result_dict = search.results(search_query)
        logger.debug(f"Search results returned for {search_query} with keys {result_dict.keys()} and values {result_dict}")
        
    except Exception as e:
        logger.error(f"Exception searching for news about {company} w/ticker {ticker} with Serper: {e}")
        
    return result_dict

def get_news_and_sentiment_from_alpha_vantage(ticker, n_search_results=3):
    articles = {}
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": config.get_api_key('alpha_vantage'),
            "limit": n_search_results
        }
        logger.debug(f"Getting news for {ticker} from Alpha Vantage")
        response = requests.get(url, params=params)
        data = response.json()

        articles['news'] = []
        logger.debug(f"Got {len(data['feed'])} news items for {ticker} from Alpha Vantage")
        for item in data['feed'][:n_search_results]:
            logger.debug(f"Processing news item: {item.keys()} from {item}")
            article = {
                "title": item['title'],
                "link": item['url'],
                "summary": item['summary'],
                "sentiment_score": item['overall_sentiment_score'],
                "sentiment_label": item['overall_sentiment_label'],
                "date": datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S').strftime('%B %d, %Y, %H:%M:%S')
            }
            logger.debug(f"Article for ticker {ticker}: {article}")
            articles['news'].append(article)
    
    except Exception as e:
        logger.error(f"Exception searching for news about ticker {ticker} with Alpha Vantage: {e}")
        
    return articles

def display_news_for_ticker(ticker):
    logger.debug(f"Displaying news for {ticker}")
    if f'{ticker}_news_and_analysis' in st.session_state:
        logger.debug(f"Displaying news for {ticker} with {len(st.session_state[f'{ticker}_news_and_analysis'])} items")
        with st.expander(f"News about {ticker}"):
            for news_item in st.session_state[f'{ticker}_news_and_analysis']:
                logger.debug(f"Displaying news item for {ticker}: {news_item} with keys {news_item.keys()}")
                
                st.markdown(f"{news_item['title']} from {news_item['date']}")
                if 'summary' in news_item:
                    st.markdown(f"{news_item['summary']}")
                
                if 'Error while summarizing' in news_item['ai_summary'] or 'No summary available' in news_item['ai_summary']:
                    st.markdown(f"OpenAI Summary: {news_item['ai_summary']}")
                else:
                    st.success(f"OpenAI Summary: {news_item['ai_summary']}")
                
                if 'sentiment_label' in news_item and news_item['sentiment_label'] and 'sentiment_score' in news_item and news_item['sentiment_score']:
                    st.markdown(f"Sentiment: {news_item['sentiment_label']} ({news_item['sentiment_score']})")
                st.markdown(f"[Read more]({news_item['link']})")
    else:
        st.markdown(f"No news found for {ticker}")
        
def get_company_name(symbol):
    url = "https://www.alphavantage.co/query"
            
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": symbol,
        "apikey": config.get_api_key('alpha_vantage'),
    }
    response = requests.get(url, params=params)
    json_response = response.json()
    
    if 'bestMatches' not in json_response:
        print(f"Error fetching data for symbol {symbol}: {json_response}")
        return None

    for item in json_response['bestMatches']:
        if item['1. symbol'] == symbol:
            return item['2. name']

    return None