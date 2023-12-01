# pip install openai chromadb python-dotenv bs4 argparse lxml
import logging
import argparse
import asyncio
from bs4 import BeautifulSoup
import chromadb
from chromadb.db.base import UniqueConstraintError  # Import the exception
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json
import openai
from openai import OpenAI
import os
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level to INFO

# Load environment variables
load_dotenv()

# Set your OpenAI key and configure OpenAI Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Embeddings functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

# Initialize ChromaDB Client
# chroma_client = chromadb.Client() # in-memory db
chroma_client = chromadb.PersistentClient(path="db") # persistent db
collection_name = "sitemap_collection"

async def fetch_sitemap(url):
    """
    Asynchronously fetch a sitemap from the given URL.
    Returns the sitemap's XML content as a string, or None if an error occurs.
    """
    try:
        response = await asyncio.to_thread(requests.get, url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching sitemap: {e}")
        return None

def parse_sitemap(sitemap_content, max_urls=None):
    """
    Parse the sitemap content and extract a limited number of URLs.

    Args:
    sitemap_content (str): XML content of the sitemap.
    max_urls (int, optional): Maximum number of URLs to extract. If None, extracts all URLs.

    Returns:
    List[str]: A list of extracted URLs, limited to 'max_urls' if specified.
    """
    import xml.etree.ElementTree as ET

    # Parse the XML content
    tree = ET.ElementTree(ET.fromstring(sitemap_content))
    root = tree.getroot()

    # Define the namespace map for parsing sitemap XML
    # Adjusted to handle both http and https namespaces
    namespaces = {
        'http': 'http://www.sitemaps.org/schemas/sitemap/0.9',
        'https': 'https://www.sitemaps.org/schemas/sitemap/0.9'
    }

    # Try to extract the URLs with http namespace first
    urls = [element.text for element in root.findall('.//http:loc', namespaces)]
    
    # If no URLs found, try with https namespace
    if not urls:
        urls = [element.text for element in root.findall('.//https:loc', namespaces)]

    # Limit the number of URLs if max_urls is specified
    if max_urls is not None:
        urls = urls[:max_urls]

    return urls

async def fetch_and_save_html(url, update_progress, collection):
    """
    Fetch the HTML content of a given URL, extract and clean text from main content elements, 
    and save it to ChromaDB.
    """
    try:
        response = await asyncio.to_thread(requests.get, url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.extract()
        
        # Extract main content using common content markers
        main_content = soup.find_all(['article', 'main', 'div'], class_=lambda x: x and 'content' in x)
        
        # If no common content markers found, fall back to extracting all text
        if not main_content:
            main_content = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        
        text_content = ' '.join(element.get_text(strip=True, separator=' ') for element in main_content)

        save_to_chromadb(url, text_content, collection)
        update_progress()
    except requests.RequestException as e:
        logging.error(f"Error fetching HTML content from {url}: {e}")


def save_to_chromadb(url, html_content, collection):
    """
    Save the HTML content to ChromaDB with detailed error handling.
    """
    try:
        collection.upsert(
            documents=[html_content],
            metadatas=[{"url": url}],
            ids=[url]
        )
    except UniqueConstraintError as e:
        logging.warning(f"Duplicate entry for {url} not added to ChromaDB: {e}")
    except Exception as e:
        # Log the exception type and message
        logging.error(f"Exception while adding/updating {url} in ChromaDB: {type(e).__name__}, {e}")

        # Optionally, log a snippet of the content for further inspection
        snippet = html_content[:200]  # Adjust the length as needed
        logging.info(f"Content snippet: {snippet}")

def search_in_chromadb(query, n_results, collection):
    """
    Search in ChromaDB for the given query.

    Args:
    query (str): The search query.
    n_results (int): Number of search results to return.

    Returns:
    List of search results.
    """

    # Search in the collection
    search_results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return search_results

def write_article(prompt):
    """
    Generate an article using the OpenAI API.
    """
    try:
        response = openai_client.chat.completions.create(
            model='gpt-4-1106-preview',
            messages=[
                {"role": "system", "content": "Follow user instructions. Write using Markdown."},
                {"role": "user", "content": prompt}
            ]
        )
        # Access the 'content' attribute of the last message in the response
        last_message_content = response.choices[0].message.content
        return last_message_content

    except Exception as e:
        print(f"An error occurred during article generation: {e}")
        return None