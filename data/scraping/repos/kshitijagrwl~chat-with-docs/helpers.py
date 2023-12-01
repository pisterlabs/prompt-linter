import requests
import json
from aiohttp import ClientSession
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from duckduckgo_search import DDGS
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from twisted.internet import reactor
from docCrawler.docCrawler.spiders.spider import DocSpider  # Replace 'your_project_name' with your actual project name

def start_crawler(start_url,max_depth=1):
    """
    Start the crawler asynchronously using Twisted with given parameters.

    Parameters:
        start_url (str): The url you want as a starting point for your docs.
        allowed_domain (str): The domain you want to crawl.
        max_depth (int): The maximum depth you want to crawl.

    Returns:
        twisted.internet.defer.Deferred: A deferred object representing the asynchronous crawling process.
    """
    # Configure logging
    configure_logging()

    # Create a CrawlerRunner
    runner = CrawlerRunner()

    # Start the spider with the given parameters
    deferred = runner.crawl(DocSpider, start_url = start_url,max_depth=max_depth)

    # Add a callback to stop the Twisted reactor once crawling is completed
    deferred.addCallback(stop_reactor)

    return deferred

def stop_reactor(_):
    """
    Callback function to stop the Twisted reactor when crawling is completed.

    Parameters:
        _: This parameter is not used.

    Returns:
        None
    """
    reactor.stop()


MAX_CONCURRENT_REQUESTS = 5  # Adjust as needed

async def _fetch_article_summary(semaphore, session, url):
    async with semaphore:
        try:
            async with session.get(url) as response:
                html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            # Find all the text on the page
            text = soup.get_text()
        except Exception as e:
            print(f"Failed to fetch article summary for {url}: {str(e)}")
            return None
        return text

async def _process_links(semaphore, links):
    async with ClientSession() as session:
        tasks = []
        for link in links:
            try:
                task = asyncio.create_task(_fetch_article_summary(semaphore, session, link))
                tasks.append(task)
            except Exception as e:
                print(f"Failed to generate summary for {link}: {str(e)}")

        return await asyncio.gather(*tasks)

def get_full_text(path='./files/output_links.json'):
    with open(path, 'r') as json_file:
        crawled_data = json.load(json_file)

    summary_data_final = {}
    summary_data = []
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    for url in crawled_data:
        links = crawled_data[url]
        
        # Process links asynchronously with semaphore
        results = asyncio.run(_process_links(semaphore, links))
        
        for text in results:
            if text:
                summary_data.append({'text': text})
    
    summary_data_final['chatwithdocs'] = tuple(summary_data)

    with open('./files/summary_output.json', 'w') as json_file:
        json.dump(summary_data_final, json_file, indent=4)
        
def split_docs(documents,chunk_size=500,chunk_overlap=20):
    """
    Split a list of documents into smaller chunks of text.

    Parameters:
        documents (list): A list of text documents.
        chunk_size (int): Maximum size of each chunk (default: 500).
        chunk_overlap (int): Number of characters to overlap between adjacent chunks (default: 20).

    Returns:
        list: A list of split documents.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def data_loader(path = './files/summary_output.json',jq_schema = '.chatwithdocs[].text'):
    """
    Load text data from a JSON file using a JSON Loader.

    Parameters:
        path (str): Path to the JSON file containing the data.

    Returns:
        list: A list containing the loaded text data.
    """
        
    loader = JSONLoader(file_path = path,jq_schema = jq_schema)
    data = loader.load()
    return data

def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        
def init_database(docs, model_name = "all-MiniLM-L6-v2"):
    """
    Initialize a vector database using the provided documents and model.

    Parameters:
        docs (list): A list of text documents.
        model_name (str): Name of the Sentence Transformer model (default: "all-MiniLM-L6-v2").

    Returns:
        Chroma: The initialized text database.
    """    
    embedding = SentenceTransformerEmbeddings(model_name = model_name)
    split_docs_chunked = split_list(docs, 41000)

    for split_docs_chunk in split_docs_chunked:
        vectordb = Chroma.from_documents(
            documents=split_docs_chunk,
            embedding=embedding,
            persist_directory="./files/chroma_db",
        )
        vectordb.persist()

def search(query):
    """
    Searches for the given query using the DuckDuckGo search engine.

    Args:
        query (str): The search query.

    Returns:
        str: The first result returned by the search engine.
    """
    with DDGS() as ddgs:
        for r in ddgs.text(query):
            return r

def search_similarity(query,openai_api_key="your openai api key", embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), openai_model="gpt-3.5-turbo-16k"):
    """
    Perform a similarity search on the text database using the given query.

    Parameters:
        query (str): The query for the similarity search.
        embeddings (SentenceTransformerEmbeddings): An instance of SentenceTransformerEmbeddings
            with the desired model for generating text embeddings.
        openai_api_key (str): Your OpenAI API key for using the OpenAI language model.
        openai_model (str): The name of the OpenAI language model to use.

    Returns:
        str: The search result as an answer.
    """
    
    database = Chroma(persist_directory="./files/chroma_db", embedding_function=embeddings)
    query = query.lower()
    
    # matching_docs = database.similarity_search(query)
    tools = [
    Tool(
        name="simsearch",
        func=database.similarity_search,
        description="useful for when you need to search documentation for a specific topic using vector database,use the query parameter to specify the input",
    ),
    Tool(
        name="search",
        func=search,
        description="useful for when you don't get the answer you want from the similarity search,use the query parameter to specify the input",
    ), 
]
    model_name = openai_model
    llm = ChatOpenAI(model_name=model_name,openai_api_key = openai_api_key)
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
    )
    answer =  agent.run("Imagine you are a software developer, you have to help the user with their query as best as you can. When you find multiple documents from simsearch use all of them to generate an answer yourself. Write proper descriptions for your answer and give code if necessary. Use the information from the similarity search to help answer the query. If you find any blog or article you can use webcrawl, webcrawl does not support pdf. Use all tools available to make your answer. Query: "+query)
    return answer