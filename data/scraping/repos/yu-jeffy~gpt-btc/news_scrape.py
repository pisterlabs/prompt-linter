import os
import requests
import time
from datetime import datetime
from newspaper import Article
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to fetch news articles
def fetch_news_articles():
    api_token = os.getenv('CRYPTONEWS_API_TOKEN')
    url = f"https://cryptonews-api.com/api/v1?tickers=BTC&items=30&page=1&token={api_token}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['data']
    else:
        print("Failed to fetch news articles")
        return []

# Variable holding 30 articles
articles = fetch_news_articles()
print(articles)

def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to scrape {url}: {str(e)}")
        return None

# Function to scrape all articles and store them
def scrape_all_articles(articles):
    scraped_articles = []
    for article in articles:
        article_content = scrape_article(article['news_url'])
        if article_content:
            scraped_articles.append({
                'title': article['title'],
                'text': article_content,
                'source_name': article['source_name'],
                'date': article['date'],
                'sentiment': article['sentiment'],  # Provided by Cryptonews API
                'url': article['news_url']
            })
    return scraped_articles

# Scrape all 30 articles using Newspaper3k
scraped_articles = scrape_all_articles(articles)

# Print a snippet of each scraped article's details
for article in scraped_articles:
    snippet = (article['text'][:200] + '...') if len(article['text']) > 200 else article['text']
    print(f"Title: {article['title']}")
    print(f"Source: {article['source_name']}")
    print(f"Date: {article['date']}")
    print(f"Sentiment: {article['sentiment']}")
    print(f"URL: {article['url']}")
    print(f"Content Snippet:\n{snippet}\n")
    print("------------------------------------------------\n")



# Function to analyze an article with GPT-4
def analyze_article_with_gpt(article_text):
    try:
        # System prompt
        system_prompt = (
        "You are a highly knowledgeable assistant with expertise in Bitcoin and cryptocurrency markets. "
        "Your task is to analyze news articles related to Bitcoin. For each article, you will provide a concise summary. "
        "After summarizing, you will rate the sentiment of the article, its relevance to the Bitcoin and cryptocurrency markets, "
        "and its overall importance. Rate each of these three aspects on a scale from 0 to 100, where 0 is the lowest and 100 is the highest. "
        "Your responses should be factual, unbiased, and based solely on the content of the article. "
        "Respond in a structured format that includes the summary followed by the ratings for sentiment, relevance, and importance. "
        "For example: 'Summary: [Your summary here]. Sentiment: [0-100], Relevance: [0-100], Importance: [0-100].' "
        "Do not have any text following the integer rating for Sentiment, Relevance, nor Importance."
        "Avoid speculation and provide analysis based on the information available in the article."
        )

        # Structured prompt for analysis
        analysis_prompt = (
            f"Analyze the following Bitcoin-related article and provide a summary, "
            f"then rate its sentiment, market relevance, and importance on a scale from 0 to 100. "
            f"Do not have any text following the integer rating for Sentiment, Relevance, nor Importance."
            f"We will be parsing the response with a predetermined format, STRICTLY follow the format"
            f"Respond in the exact following format: "
            f"Summary: [Your summary here] "
            f"Sentiment: [0-100], Relevance: [0-100], Importance: [0-100].\n\n{article_text}"
        )

        # Create the chat completion
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        # Assuming the response follows the format, we extract the summary and ratings
        response_content = response.choices[0].message.content
        # print(response.choices[0].message.content)
        summary = response_content.split('Summary: ')[1].split(' Sentiment: ')[0].strip()
        
        # Extracting sentiment, relevance, and importance, handling extra text
        sentiment_str = response_content.split('Sentiment: ')[1].split(',')[0].split(' ')[0].strip()
        relevance_str = response_content.split('Relevance: ')[1].split(',')[0].split(' ')[0].strip()
        importance_str = response_content.split('Importance: ')[1].split('.')[0].split(' ')[0].strip()

        # Converting string to integer, handling any non-numeric characters
        sentiment = int(''.join(filter(str.isdigit, sentiment_str)))
        relevance = int(''.join(filter(str.isdigit, relevance_str)))
        importance = int(''.join(filter(str.isdigit, importance_str)))

        return {
            'summary': summary,
            'sentiment': sentiment,
            'relevance': relevance,
            'importance': importance
        }
    except Exception as e:  # Catching a general exception for simplicity
        print(f"An error occurred: {str(e)}")
        return None

# Function to analyze and store the results for all articles
def analyze_and_store_articles(articles):
    for article in articles:
        analysis_results = analyze_article_with_gpt(article['text'])
        if analysis_results:
            article.update(analysis_results)

# Analyze all articles once and store the results
analyze_and_store_articles(scraped_articles)

# Print the summaries and ratings from the stored results
for article in scraped_articles:
    print(f"Title: {article['title']}")
    print(f"Summary: {article['summary']}")
    print(f"Sentiment Rating: {article['sentiment']}")
    print(f"Market Relevance Rating: {article['relevance']}")
    print(f"Importance Rating: {article['importance']}")
    print("------------------------------------------------\n")

# Function to save results to a file with a timestamp in the filename
def save_results_to_file(articles, results_directory='results'):
    # Ensure the results directory exists
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{results_directory}/news_analysis_{timestamp}.txt"
    
    with open(filename, 'w') as file:
        for article in articles:
            file.write(f"Title: {article['title']}\n")
            file.write(f"Summary: {article['summary']}\n")
            file.write(f"Sentiment Rating: {article['sentiment']}\n")
            file.write(f"Market Relevance Rating: {article['relevance']}\n")
            file.write(f"Importance Rating: {article['importance']}\n")
            file.write("------------------------------------------------\n\n")
    print(f"Analysis results saved to {filename}")

# Save the results to a file with a timestamped filename
save_results_to_file(scraped_articles)

def main():
    articles = fetch_news_articles()
    print(articles)
    scraped_articles = scrape_all_articles(articles)
    analyze_and_store_articles(scraped_articles)
    save_results_to_file(scraped_articles)

if __name__ == '__main__':
    main()