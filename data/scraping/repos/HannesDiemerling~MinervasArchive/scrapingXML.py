from urllib.parse import urlparse
import scrapingFunctions as scraping
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import embed
from langchain.vectorstores import Chroma
import pandas as pd

# initializing auth.py
exec(open('auth.py').read())

# URL of the main sitemap for report
url = "https://www.rr23.mpib-berlin.mpg.de/sitemap.xml"
# Fetch all urls of the report
df = scraping.get_all_urls(url)

# Create a WebBaseLoader object and load the document
report_docs = embed.webpages(df)

Chroma.from_documents(report_docs, embed.embedding(), persist_directory=embed.report_db_dir())

# Fetch urls for persons

base_url = "https://www.mpib-berlin.mpg.de/person/"
start_url = "https://www.mpib-berlin.mpg.de/research/research-centers/adaptive-rationality/people"

urls = scraping.scrape_urls(base_url, start_url)

# Fetch for Staff

base_url = "https://www.mpib-berlin.mpg.de/staff"
start_url = "https://www.mpib-berlin.mpg.de/staff"
startswith="/staff/"

urls2 = scraping.scrape_urls(base_url, start_url,startswith=startswith)
urls=urls+urls2

for i in range(1,22):
    print(start_url+"?page="+str(i+1))
    urls2 = scraping.scrape_urls(base_url, start_url+"?page="+str(i+1),startswith=startswith)
    urls=urls+urls2

df = scraping.urls_to_dataframe(urls, start_url)

# Create a WebBaseLoader object and load the document
person_docs = embed.webpages(df)

Chroma.from_documents(person_docs, embed.embedding(), persist_directory=embed.person_db_dir())

