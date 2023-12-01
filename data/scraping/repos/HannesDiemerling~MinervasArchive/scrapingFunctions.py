import pandas as pd
import urllib.request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import re
from langchain.schema import Document
import os
from dotenv import load_dotenv
import numpy as np

# Function to fetch the raw XML of the sitemap
def get_sitemap(url):
    response = urllib.request.urlopen(url)
    xml = BeautifulSoup(response, 'lxml-xml', from_encoding=response.info().get_param('charset'))
    return xml

# Function to determine the type of sitemap
def get_sitemap_type(xml):
    sitemapindex = xml.find_all('sitemapindex')
    sitemap = xml.find_all('urlset')

    if sitemapindex:
        return 'sitemapindex'
    elif sitemap:
        return 'urlset'
    else:
        return

# indexieren der personen auf namen
# Function to fetch the URLs of the child sitemaps
def get_child_sitemaps(xml):
    sitemaps = xml.find_all("sitemap")
    output = []
    for sitemap in sitemaps:
        output.append(sitemap.findNext("loc").text)
    return output

# Function to parse the sitemap.xml file into a pandas DataFrame
def sitemap_to_dataframe(xml, name=None):
    df = pd.DataFrame(columns=['loc', 'changefreq', 'priority', 'domain', 'sitemap_name'])
    urls = xml.find_all("url")

    for url in urls:
        if xml.find("loc"):
            loc = url.findNext("loc").text
            parsed_uri = urlparse(loc)
            domain = '{uri.netloc}'.format(uri=parsed_uri)
        else:
            loc = ''
            domain = ''

        if xml.find("changefreq"):
            changefreq = url.findNext("changefreq").text
        else:
            changefreq = ''

        if xml.find("priority"):
            priority = url.findNext("priority").text
        else:
            priority = ''

        sitemap_name = name if name else ''

        row = {
            'domain': domain,
            'loc': loc,
            'changefreq': changefreq,
            'priority': priority,
            'sitemap_name': sitemap_name,
        }
        #print(type(df)) 
        row_df = pd.DataFrame([row])  # Create a DataFrame from the row
        df = pd.concat([df, row_df], ignore_index=True)
    return df

# Function to fetch all URLs from a site's XML sitemaps
def get_all_urls(url):
    xml = get_sitemap(url)
    sitemap_type = get_sitemap_type(xml)

    if sitemap_type =='sitemapindex':
        sitemaps = get_child_sitemaps(xml)
    else:
        sitemaps = [url]

    df = pd.DataFrame(columns=['loc', 'changefreq', 'priority', 'domain', 'sitemap_name'])

    for sitemap in sitemaps:
        sitemap_xml = get_sitemap(sitemap)
        df_sitemap = sitemap_to_dataframe(sitemap_xml, name=sitemap)
        df = pd.concat([df, df_sitemap], ignore_index=True)

    return df

def get_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

class ReportWebpage:
    def __init__(self,title, content, url):
        self.title = title
        self.page_content = content
        self.url = url

def scrape_urls(base_url, start_url, startswith='/person/'):
    response = requests.get(start_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        url = link.get('href')
        if url and url.startswith(startswith):
            full_url = urllib.parse.urljoin(base_url, url)
            urls.append(full_url)

    return urls

def urls_to_dataframe(urls, start_url):
    df = pd.DataFrame(columns=['loc', 'changefreq', 'priority', 'domain', 'sitemap_name'])
    for url in urls:
        new_row = pd.DataFrame({'loc': [url], 
                                'changefreq': [np.nan], 
                                'priority': [np.nan], 
                                'domain': [start_url], 
                                'sitemap_name': [np.nan]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df
