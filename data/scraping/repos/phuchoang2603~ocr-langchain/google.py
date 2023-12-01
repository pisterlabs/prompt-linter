import requests
import urllib.parse
import json
# Web scraping
from bs4 import BeautifulSoup
# from langchain.document_loaders import AsyncHtmlLoader
# from langchain.document_transformers import Html2TextTransformer, BeautifulSoupTransformer
# from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API")
GOOGLE_SEARCH_ID = os.getenv("GOOGLE_SEARCH_ID")

def search_product_links (query):
    """Tìm kiếm các link giúp hỗ trợ phân biệt hàng thật hàng giả của một sản phẩm"""
    
    data = {
    "key": GOOGLE_API,
    "cx": GOOGLE_SEARCH_ID,
    "q": query
    }

    # Convert data to a query string and encode it
    query_string = urllib.parse.urlencode(data)
    url = "https://www.googleapis.com/customsearch/v1?" + query_string

    # Make the HTTP request and get url to provide the model
    response = requests.get(url)
    data = response.json()["items"]

    # Return list of urls in json format
    output = []
    
    for item in data:
        title = item["title"]
        link = item["link"]
        provider = item["displayLink"]
        output.append({
            "title": title,
            "link": link,
            "provider": provider
        })

    print ("Done searching")

    # write to json file
    with open('./output/search-result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    return output

# def run_playwright(site):
#     """Trích xuất thông tin từ link hướng dẫn phân biệt hàng thật hàng giả của một sản phẩm"""

#     data = ""
#     print (f"Processing {site}...")

#     try:
#         # Run playwright
#         urls = [site]
#         loader = AsyncHtmlLoader(urls)
#         docs = loader.load()

#         # Extract text from html
#         bs_transformer = BeautifulSoupTransformer()
#         docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["article"])
#         data = docs_transformed[0].page_content
#         return data

#     except:
#         print ("Error getting information from " + site)
#         print ("Attempt to get information from next site")
#         return "error"
        
# def get_search_result(product):
#     # clear output.txt
#     with open('./output/output.md', 'w', encoding='utf-8') as f:
#         f.write("")
    
#     query = "làm thế nào để phân biệt hàng thật hàng giả của " + product + "?"
#     search_results = search_product_links(query)

#     for item in search_results:
#         content = run_playwright(item["link"])

#         if content != "error":
#             with open('./output/output.md', 'a', encoding='utf-8') as f:
#                 f.write(content)
#             break
#         else:
#             continue