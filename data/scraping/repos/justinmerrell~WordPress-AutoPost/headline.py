import os
import random
import requests
import openai
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_real_headlines(api_key=NEWSAPI_KEY):
    '''
    Fetches the latest headlines related to AI, tech, entrepreneurship, and self-help.
    '''
    url = "https://newsapi.org/v2/everything"

    topics_of_interest = [
        "artificial intelligence OR machine learning OR deep learning",
        "technology trends OR tech innovations OR emerging technologies",
        "entrepreneurship OR startups OR business growth OR venture capital",
        "self-help OR personal development OR productivity OR time management",
        "blockchain OR cryptocurrency OR Bitcoin OR Ethereum",
        "internet of things OR IoT OR smart devices OR connected devices",
        "augmented reality OR virtual reality OR AR OR VR",
        "sustainable tech OR green technology OR renewable energy OR clean tech",
        "biotech OR genomics OR CRISPR OR bioinformatics",
        "data science OR big data OR data analytics OR data visualization",
        "digital marketing OR SEO OR social media trends OR content strategies",
        "fintech OR financial technology OR digital banking OR mobile payments",
        "e-commerce OR online shopping trends OR digital retail OR dropshipping",
        "future of work OR remote work OR digital nomads OR gig economy",
        "healthtech OR digital health OR telemedicine OR wearables",
        "education technology OR edtech OR online learning OR e-learning",
        "space tech OR space exploration OR SpaceX OR NASA",
        "robotics OR automation OR industrial robots OR drones",
        "cybersecurity OR cyber threats OR hacking OR online privacy",
        "design thinking OR UX design OR UI trends OR web design"
    ]

    params = {
        "q": topics_of_interest[random.randint(0, len(topics_of_interest) - 1)],
        "apiKey": api_key,
        "pageSize": 10,
        "page": random.randint(1, 5),
        "sortBy": "publishedAt",
        "language": "en",
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()
        if data["totalResults"] > 0:
            headlines = [article["title"] for article in data["articles"]]
            return headlines
        else:
            return ["No articles found related to AI or the latest tech trend."]
    else:
        raise Exception(f"Error: Unable to fetch articles. Status code: {response.status_code}")


def paraphrase_headline(headlines):
    """
    Generates a new blog post title based on the provided headlines.
    """
    with open("src/prompts/headline.txt", "r", encoding="UTF-8") as headline_prompt_file:
        prompt = headline_prompt_file.read()

    prompt = prompt.replace("{{HEADLINES}}", str(headlines))

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip(), prompt


def blog_post_title():
    '''
    Generates a new blog post title based on the latest news headlines.
    '''
    proposed_headline, prompt = paraphrase_headline(get_real_headlines())

    # Strip quotes from the headline
    proposed_headline = proposed_headline.replace('"', '')
    prompt = prompt.replace("'", "")

    return proposed_headline, prompt


if __name__ == "__main__":
    original_headlines = get_real_headlines()
    new_headline, _ = paraphrase_headline(original_headlines)

    print(f"Original Headlines: {original_headlines}")
    print(f"New Headline: {new_headline}")
