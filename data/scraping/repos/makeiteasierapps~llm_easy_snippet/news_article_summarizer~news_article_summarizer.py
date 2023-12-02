# Import necessary libraries
import os
from dotenv import load_dotenv
import requests
from newspaper import Article
from langchain.schema import (
    HumanMessage
)
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
os.getenv('OPENAI_API_KEY')
apikey = os.getenv('GNEWS_API_KEY')

# Fetch article URLs based on query
def get_article_urls(query):
    # Construct API URL
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=us&max=10&apikey={apikey}"

    try:
        articles = requests.get(url, timeout=10)
    except Exception as exception:
        print(f"Error occurred while fetching articles: {exception}")
        return

    if articles.status_code != 200:
        print(f"Request failed with status code: {articles.status_code}")
        return

    data = articles.json()
    articles = data["articles"]
    article_urls = [article_data["url"] for article_data in articles]

    return summarize_articles(article_urls)


# Summarize articles
def summarize_articles(article_urls):
    summarized_articles = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    session = requests.Session()

    for article_url in article_urls:
        try:
            response = session.get(article_url, headers=headers, timeout=10)
            # The free version of the GNews API only returns a part of the article so to create the summary
            # we scrape the article using the newspaper library. Some sites may block this, if so we just skip
            article = Article(article_url)
            article.download()
            article.parse()
        except Exception as exception:
            print(f"Error occurred while fetching article at {article_url}: {exception}")
            continue

        if response.status_code != 200:
            print(f"Failed to fetch article at {article_url}")
            continue



        # Extract article data
        article_title = article.title
        article_text = article.text

        # Prepare prompt template
        template = """You are a very good assistant that summarizes online articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        Write a summary of the previous article.
        """

        prompt = template.format(article_title=article_title, article_text=article_text)

        messages = [HumanMessage(content=prompt)]

        chat = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

        # Generate summary using chat model
        summary = chat(messages)

        # Create article dictionary
        article_dict = {
            'title': article_title,
            'summary': summary.content,
            'url': article_url
        }

        summarized_articles.append(article_dict)

    return summarized_articles


# Fetch article URLs for query 'AI' and print summarized articles
print(get_article_urls('AI'))
