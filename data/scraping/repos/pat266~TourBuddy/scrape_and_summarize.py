from duckduckgo_search import DDGS
from openai import OpenAI
from config import Config
from scraper import Scraper

class ScrapeAndSummarize():
    def __init__(self):
        self.ddgs = DDGS()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.scraper = Scraper()

    def ddgsearch(self, query, numresults=10):
        query = query + ' tripadvisor -site:facebook.com -site:yelp.com'
        results = list(self.ddgs.text(query, max_results=numresults))
        urls = [result['href'] for result in results][:numresults]
        print(urls)
        crawled_reviews = self.scraper.run(urls)
        return ' '.join([review['raw_content'] for review in crawled_reviews])

    def summarize_reviews(self, all_reviews, place_type):
        prompt = f"""
        "From the following paragraph about a place, please identify and summarize the key details regarding:
        1) the estimated cost of entry (preferably in number).
        2) the most popular activities, or if this is a restaurant, the most popular dishes or what people commonly order.
        and 3) the environment and atmosphere of the place.
        {all_reviews}"
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            max_tokens=500,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.choices[0].message.content.strip()
