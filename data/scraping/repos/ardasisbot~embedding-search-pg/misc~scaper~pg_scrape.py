# script to scrape PG's essays, call OpenAI embeddings and store it in database

import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import SpacyTextSplitter
from dbutils import initialize_supabase_client, insert_text_metadata_to_db
from dotenv import load_dotenv

# makes sure OpenAIEmbeddings sees our OpenAI api key
load_dotenv()

def extract_date_from_text(text):
  
  import re

  # define the regex pattern to match the date
  pattern = r'([A-Z][a-z]+) (\d{4})'

  # search for the date pattern in the HTML content
  pattern_match = re.search(pattern, text)

  if pattern_match:
      # extract the matched groups (month and year)
      month = pattern_match.group(1)
      year = pattern_match.group(2)
      # print the extracted date
    #   print(f"{month} {year}")
      return month, year
      
  else:
      print("Date not found.")
  return None, None

article_url = "http://www.paulgraham.com/read.html"

def scrape_pg_article(article_url):
    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = httpx.get(article_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    import re
    # extract title
    article_title = soup.find('title').string
    article_month, article_year = extract_date_from_text(soup.text)

    # look at all the text came after the year
    chunks = soup.text.split(article_year)[1]

    # filter out the notes
    notes_match = re.search(r'Notes?\s*\[', chunks)  # search for the pattern in the HTML content
    if notes_match:
        text_body = chunks[:notes_match.start()] 
    else: 
        text_body = chunks
    

    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap = 100)

    text_chunks = text_splitter.split_text(text_body)
        
    return [{'essay_title': article_title,
            'essay_url': article_url, 
            'essay_date': article_year + ' ' + article_month,
            'content': text_chunk, 
            'content_length': len(text_chunk)} for text_chunk in text_chunks]


def get_pg_article_urls():
    import httpx
    from bs4 import BeautifulSoup

    # Fetch the HTML content of the web page
    url = "http://www.paulgraham.com/articles.html"
    response = httpx.get(url)
    html_content = response.text

    # Use BeautifulSoup to parse the HTML content and extract the links
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href is not None and href.endswith('.html'):
            links.append(href)

    links = links[1:-1]

    # append paulgraham.com to the URL 
    links = [f'http://www.paulgraham.com/{link}' for link in links]

    return links

def get_embeddings(text_str):
    from langchain.embeddings.openai import OpenAIEmbeddings
    try: 
        embedding = OpenAIEmbeddings().embed_query(text_str)
    except Exception as e: 
        print('OpenAI embedding exception {e}')
    return embedding

if __name__ == "__main__":
    pg_article_urls = get_pg_article_urls()

    for article_url in pg_article_urls[:-5]:
        # note: can be asnyc (modal has an example on this)
        # note: thought about splitting scraping & embedding, but supabase python-client doesnt allow update
        try: 
            metadata_list = scrape_pg_article(article_url)

            for metadata in metadata_list:
                # pull in OpenAI embeddings
                metadata['embedding'] = get_embeddings(metadata['content'])
            
            supabase_client = initialize_supabase_client()
            insert_text_metadata_to_db(supabase_client, metadata_list)
        except Exception as e: 
            print(f'{article_url}: Throwing exception {e}')

