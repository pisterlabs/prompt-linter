from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time
import sys
import openai
import os
import json
import datetime

from gpt_translate.articles.JsonArticleManager import JsonArticleManager
from gpt_translate.crawl.util import extract_info, scroll_one_step
import pandas as pd

import logging

LOG_FILE_PATH = "/opt/shichenh/blogger_crawl.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("crawl netease")

ARTICLE_JSON_PATH = "/opt/shichenh/articles_embedding.json"
article_manager = JsonArticleManager(ARTICLE_JSON_PATH)


#num_articles = ...  # Set the desired number of articles
url = "https://m.163.com/news/sub/T1658526449605.html"

# Set up headless browser
options = Options()
options.add_argument('--headless')
#options.headless = True
driver = webdriver.Chrome(options=options)

# Load webpage and extract data
driver.get(url)



# scroll until getting all pages
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait for the page to load
    time.sleep(2)

    # Get new page height after scrolling
    new_height = driver.execute_script("return document.body.scrollHeight")

    # If the page height has not changed, we've reached the end of the page
    if new_height == last_height:
        break
    last_height = new_height

# Parse the page source with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find all elements with the specified class
elements = soup.find_all("li", class_="single-picture-news js-click-news")
logger.info(f"num_elements: {len(elements)} crawled")

articles_info = pd.DataFrame(extract_info(elements)).drop_duplicates(['title'])
logger.info(f"number of articles found: {articles_info.shape[0]}")

existing_titles = set(article_manager.articles_df['title'])
new_articles_df = articles_info[~articles_info['title'].isin(existing_titles)]


##### crawl actual page information
logger.info(f"number of new articles: {new_articles_df.shape[0]}")
logger.info("crawl actual page information")
url_to_dictionary = {}
if new_articles_df.shape[0]:
    urls = new_articles_df['link'].apply(lambda url: "https://" + url.replace("//","")).tolist()
    for url in urls:
        try:
            driver.get(url)
            
            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Find all elements with the specified class

            content_div = soup.find("section", class_="article-body js-article-body")
            element_text = content_div.get_text()
            reformmated_text = element_text.replace("。", "\n")
            
            url_to_dictionary[url] = reformmated_text
        except Exception as e:
            logger.error(f"error processing {url}")
            logger.error(e)
else:
    sys.exit()


new_articles_df['url'] = new_articles_df['link'].apply(lambda url: "https://" + url.replace("//",""))

new_articles_df['text'] = new_articles_df['link'].apply(lambda url: "https://" + url.replace("//","")).apply(lambda url: url_to_dictionary.get(url, "")).\
    apply(lambda text:
         text.replace("文章来自微信公众号：记忆承载\n欢迎前往关注阅读全文\n", "").\
          replace("\n文章来自微信公众号：记忆承载\n欢迎前往关注阅读全文\n\n\n\n\n打开网易新闻 查看更多图片", "")
         )

new_articles_df.to_csv("./new_articles_df.csv", index=False)
today_str = str(datetime.datetime.now().date())
new_articles_df['date'] = today_str

records = new_articles_df.to_dict("records")

logger.info("running the translation pipeline")
openai.api_key = os.environ["OPENAI_API_KEY"]
completion_config = {
    'model': "gpt-3.5-turbo-16k",
    'temperature': 0
}


for record in records:
    try:
        if record['text']:
            logger.info(f"processing record: {record['title']}")
            record['text'] = record['text'].replace("\n\n","\n")
            new_record = article_manager.complete_article(record, completion_config)
            article_manager.add_complete_article(new_record)
        else:
            continue
    except Exception as e:
        logger.error(f"error processing record {record['title']}")
        logger.error(e)

logger.info(f"new total article size: {article_manager.articles_df.shape[0]}")

logger.info("normalize the dates")
from gpt_translate.crawl.util import normalize_date
article_manager.articles_df['date'] = article_manager.articles_df['date'].apply(normalize_date)
article_manager.articles_df.to_json(article_manager.article_json_path, orient='records')