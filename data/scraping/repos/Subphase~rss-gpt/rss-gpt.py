import feedparser
import requests
import openai
import time
from os import getenv
from bs4 import BeautifulSoup
from typing import List, Dict

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

class Article:
    def __init__(self, title:str, url:str, summary:str) -> None:
        self.title = title
        self.url = url
        self.summary = summary
        self.text = self.scrape_website()

    def __repr__(self) -> str:
        return f"{self.title} - {self.url}"
    
    # Scrape the self.url and return the text
    def scrape_website(self) -> str:
        # Send an HTTP GET request to the website
        response = requests.get(self.url)
        response.raise_for_status()  # Raise an exception if request fails

        # Create a Beautiful Soup object to parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find specific elements and extract the text
        text_elements = soup.find_all('p')

        # Clean up the extracted text, if necessary
        cleaned_text = [element.get_text().strip() for element in text_elements]
        return ' '.join(cleaned_text)
    
    def write_to_file(self) -> None:
        with open('articles.md', 'a') as f:
            f.write(f'### {self.title}  \n')
            f.write(f'{self.text}  \n\n')

# ChatGPT API query class
class Ai:
    def __init__(self) -> None:
        openai.api_key = OPENAI_API_KEY
        self.model = "gpt-3.5-turbo"
        self.sys_role = {"role": "system", "content": "You are a cybersecurity professional who is tasked with staying up to date on the latest news. You are reading the following article and want to know how important it is to your job. The most important articles have news about the latest cyber attacks, new vulnerabilities, and new tools."}

    def chat_prompt(self, prompt):
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = [
                self.sys_role,
                {"role": "user", "content": prompt}
            ]
        )
        if 'choices' in response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return ""

    def rate_importance(self, article: Article) -> int:
        prompt = f'''
            Rate the following article on a scale of 1-10, with 10 being the most important and 1 being the least important.\n
            ``` {article.text} ```\n 
            The response should ONLY include a number between 1 and 10 and nothing else'''
        response = self.chat_prompt(prompt)
        return response

def main() -> None:
    # Guard clause to ensure OPENAI_API_KEY is set
    if not OPENAI_API_KEY: raise Exception("OPENAI_API_KEY not set")

    websites = ['https://feeds.feedburner.com/TheHackersNews?format=xml']
    rss_feed = collect_rss_feed(websites)
    # limit rss feed to 6 articles for debugging
    #rss_feed = rss_feed[:1]
    articles = instantiate_articles_factory(rss_feed)
    articles = rate_importance(articles)
    # write the article to a file if they are rated above 7
    for article in articles:
        if article.importance < 8: break
        article.write_to_file()

def collect_rss_feed(feed_urls: List[str]) -> List[Dict[str, str]]:
    latest_posts = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            post = {
                'title': entry.title,
                'url': entry.link,
                'summary': entry.summary
            }
            latest_posts.append(post)
    return latest_posts
 
# function to instantiate articles from rss feed
def instantiate_articles_factory(rss_feed: List[str]) -> List[Article]:
    return [Article(title = item['title'], url = item['url'], summary = item['summary']) for item in rss_feed]

# function to sort articles by importance
def sort_articles(articles: List[Article]) -> List[Article]:
    return sorted(articles, key=lambda x: x.importance, reverse=True)

# function to rate the importance of each article (uses OpenAI API)
def rate_importance(articles: List[Article]) -> List[Article]:
    ai = Ai()
    for article in articles:
        attempt_count = 1
        try:
            article.importance = int(ai.rate_importance(article))
        except:
            # wait for 10 secs and try again
            attempt_count += 1
            print(f"Error in rating article {article.title}. Retry attempt {attempt_count}")
            if attempt_count > 3: continue
            time.sleep(10)
            article.importance = int(ai.rate_importance(article))

        time.sleep(20) # sleep for 20 seconds to avoid OpenAI API rate limit
    return sort_articles(articles)

if __name__ == '__main__':
    main()
