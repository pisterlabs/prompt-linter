import requests
from bs4 import BeautifulSoup
import html2text
from dotenv import load_dotenv
import os
import json
from urllib.parse import urljoin
import re
import html2text
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai

load_dotenv()

brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

print(brwoserless_api_key)
def scrape_website(url: str):

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url,
        "elements": [{
            "selector": "body"
        }]
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        f"https://chrome.browserless.io/scrape?token={'0af8da42-e7eb-4c99-a98e-34d924416991'}",
        headers=headers,
        data=data_json
    )

    # Check the response status code
    if response.status_code == 200:
        # Decode & Load the string as a JSON object
        result = response.content
        data_str = result.decode('utf-8')
        data_dict = json.loads(data_str)

        # Extract the HTML content from the dictionary
        html_string = data_dict['data'][0]['results'][0]['html']

        return html_string
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        

def convert_relative_urls_to_absolute_in_markdown(markdown, base_url):
    """
    Convert relative URLs to absolute URLs in the markdown.

    Args:
    - markdown (str): The markdown text.
    - base_url (str): The base URL of the original website.

    Returns:
    - str: The updated markdown text with absolute URLs.
    """
    # Replace relative URLs with absolute URLs in the markdown
    def replace_relative_url(match):
        relative_url = match.group(1)
        absolute_url = urljoin(base_url, relative_url)
        return f"]({absolute_url})"
    
    markdown = re.sub(r'\]\((/[^)]*)\)', replace_relative_url, markdown)
    return markdown

def get_base_url(url):
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def convert_html_to_markdown(html):

    # Create an html2text converter
    converter = html2text.HTML2Text()

    # Configure the converter
    converter.ignore_links = False

    # Convert the HTML to Markdown
    markdown = converter.handle(html)

    return markdown


from urllib.parse import urlparse
import requests


def is_valid_url(url):
    try:
        # Parse the URL
        parsed_url = urlparse(url)

        # Check if the scheme (protocol) and netloc (domain) are present
        if parsed_url.scheme and parsed_url.netloc:
            # Additional checks can be performed here, like checking if the domain exists
            # or if the URL responds with a valid status code.
            
            # You can use the `requests` library to check if the URL responds with a valid status code
            response = requests.head(url)
            return response.status_code < 400
        else:
            return False
    except:
        return False


def url_to_doc(url):
    # if is_valid_url(url):
    #     return False
    
    # query = "How to upload large files?"
    html = scrape_website(url)
    markdown = convert_html_to_markdown(html)
    base_url = get_base_url(url)
    markdown = convert_relative_urls_to_absolute_in_markdown(markdown, base_url)

    os.makedirs("documents", exist_ok=True)

    iteration_number = 1  

    filepath = os.path.join("documents", "iteration_1" , 'output1.md')
    with open(filepath, "w" , encoding="utf-8") as file:
        file.write(markdown)

    return True
