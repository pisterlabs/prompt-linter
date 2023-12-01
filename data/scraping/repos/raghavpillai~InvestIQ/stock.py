import yfinance as yf
import requests
import dotenv
import os
import math
import openai
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

dotenv.load_dotenv()
from sources.news import News
from sources.reddit import Reddit
from sources.youtube import Youtube

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


class Stock:
    def __init__(self, ticker):
        # Essentials
        self.ticker: str = ticker

        # Details
        self.name: str = None
        self.market_cap: float = None
        self.description: str = None
        self.similar: str = None
        self.current_price: float = None
        self.growth: str = None
        self.recommend: str = None
        self.blurb: str = None
        self.logo_url: str = None
        self.analyst_count: int = None

        # Data
        self.perception: float = None
        self.popularity: int = None
        self.overall_rating: float = None

    def create_blurb(self, stock_data: Dict[str, str]) -> str:
        # Delete to save tokens
        stuff_to_delete: List[str] = [
            "longBusinessSummary", "companyOfficers", "uuid", "messageBoardId",
            "address1", "website", "phone", "city", "state", "zip",
            "country", "industry", "gmtOffSetMilliseconds", "governanceEpochDate",
            "timeZoneFullName", "timeZoneShortName",
        ]

        for stuff in stuff_to_delete:
            if stuff in stock_data:
                del stock_data[stuff]
        
        stock_data["name"] = self.name
        
        # return "Insert blurb here"

        response: Dict[str, str] = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to take in stock data and return an smart but concise analysis on the market trends. Use and cite quantitative data to determine if the stock is worth buying or not. Every sentence should be a point backed up by data. Provide a single concise paragraph blurb of no more than 150 characters.",
                },
                {
                    "role": "user",
                    "content": str(stock_data),
                }
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

    def populate(self):
        if not self.ticker:
            print("Invalid ticker")
            return

        stock_details: Dict[str, str] = {}
        stock_data: Dict[str, str] = {}
        try:
            stock_details, stock_data = self._get_stock_info()
        except Exception:
            print("Unable to get stock info")
            return

        print(f"Retrieving stock info for {self.ticker}")
        print(stock_details)

        self.name = stock_details.get("name") or self.ticker
        self.market_cap = stock_details.get("marketcap")
        self.similar = stock_details.get("similar")
        self.logo = stock_details.get("logo")

        open_price = stock_data.get("regularMarketOpen")
        close_price = stock_data.get("previousClose")
        self.description = stock_data.get("longBusinessSummary")
        self.current_price = stock_data.get("currentPrice")
        self.growth = stock_data.get("revenueGrowth",0) * 100
        self.recommend = stock_data.get("recommendationKey", "Unknown")
        self.analyst_count = stock_data.get("numberOfAnalystOpinions", 0)

        self.blurb = self.create_blurb(stock_data)

        # twitter = Twitter()
        reddit: Reddit = Reddit()
        news: News = News()
        youtube: Youtube = Youtube()

        reddit_perception, reddit_popularity = reddit.calculate_perception(self.name)
        youtube_perception, youtube_popularity = youtube.calculate_perception(
            self.name
        )
        news_perception, news_popularity = news.calculate_perception(self.name)

        total_popularity: float = (
            (reddit_popularity + youtube_popularity + news_popularity) / 3
        )
        total_perception: float = (
            (reddit_perception + youtube_perception + news_perception) / 3
        ) + 0.2
        print(f"Perception: {total_perception}")
        print(f"Popularity: {total_popularity}")

        def apply_bias(score, bias_factor):
            return score * math.exp(bias_factor * abs(score))

        # Go higher/lower as needed
        bias_factor = 0.2

        biased_perception = apply_bias(total_perception, bias_factor)
        biased_popularity = apply_bias(total_popularity, bias_factor)

        overall_rating = (biased_perception + biased_popularity) / 2
        overall_rating = min(max(overall_rating, -0.98), 0.98) # Clamp

        def similarity_ratio(a: str, b: str) -> float:
            return SequenceMatcher(a=a.lower(), b=b.lower()).ratio()
        
        top_overall_titles: List[Tuple[str, str]] = [(title, "youtube") for title in youtube.top_titles] + \
                                                    [(title, "reddit") for title in reddit.top_titles] + \
                                                    [(title, "news") for title in news.top_titles]

        bottom_overall_titles: List[Tuple[str, str]] = [(title, "youtube") for title in youtube.bottom_titles] + \
                                                    [(title, "reddit") for title in reddit.bottom_titles] + \
                                                    [(title, "news") for title in news.bottom_titles]
        print(top_overall_titles)
        print(bottom_overall_titles)
        top_overall_titles.sort(key=lambda x: similarity_ratio(x[0], self.name), reverse=True)
        bottom_overall_titles.sort(key=lambda x: similarity_ratio(x[0], self.name), reverse=True)

        self.perception = round(total_perception * 100, 2)
        self.popularity = round(total_popularity * 100, 2)
        self.overall_rating = round(overall_rating * 100, 2)

        if self.perception > 0:
            majority_role = "positive"
            minority_role = "negative"
            majority_titles = top_overall_titles
            minority_titles = bottom_overall_titles
        else:
            majority_role = "negative"
            minority_role = "positive"
            majority_titles = bottom_overall_titles
            minority_titles = top_overall_titles

        # Select the top titles based on perception
        titles_to_show: List[Dict[str, str]] = [
            {"title": majority_titles[0][0], "source": majority_titles[0][1], "role": majority_role},
            {"title": majority_titles[1][0], "source": majority_titles[1][1], "role": majority_role},
            {"title": minority_titles[0][0], "source": minority_titles[0][1], "role": minority_role},
        ]
        self.titles = titles_to_show

    def _get_stock_info(self) -> Dict[str, str]:
        response: requests.Response = requests.get(
            f"https://api.polygon.io/v1/meta/symbols/{self.ticker}/company?apiKey={POLYGON_API_KEY}",
        )
        stock_details: Dict[str, str] = response.json()

        stock_raw: yf.Ticker = yf.Ticker(self.ticker)
        stock_data: Dict[str, str] = stock_raw.info
        return stock_details, stock_data


if __name__ == "__main__":
    stock = Stock("AAPL")
    stock.populate()