import os
import subprocess
import feedparser
import openai
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dateutil import parser

import string
import random


# Set API key either from an environment variable or edit here in the code
api_key = os.environ.get("OPENAI_API_KEY") or "your_openai_api_key_here"

# Design your personal pre-prompt. Below is a template
preprompt = """
You are an AI specialized in text analysis, especially news and semantics. Your job is to filter news based on the user's desires.

The next user wishes to filter out news that are ... (e.g. sports, clickbaity, stress-inducing, sensationalist)

You receive input in this format: "<ID> <title of the news>", then you pick news based on the previous wishes. You output in the format "<id_1> <id_2> ... <id_n>". It's of utmost importance to output exactly in the format of "<id_1> <id_2> ... <id_n>", otherwise the following software won't accept your output and terminates.

NO SPORTS NEWS!! (repeating the rule might help if the filter is not otherwise working)
"""

def generate_id():
    characters = string.ascii_uppercase + string.digits
    id = ''.join(random.choice(characters) for i in range(4))
    return id


def recent_news_filter(entry, last_n_hours=12):
    now_utc = datetime.now(timezone.utc)
    published_time = parser.parse(entry.published)
    published_time_utc = published_time.astimezone(timezone.utc)
    difference = now_utc - published_time_utc
    return difference.total_seconds() < (last_n_hours * 60 * 60)


class BaseNewsFetcher(ABC):
    def __init__(self):
        self.news_feed = feedparser.parse(self.feed_source)

    def get_news(self):
        news = []
        for entry in self.news_feed.entries:
            if self.pre_filter(entry):
                news.append((generate_id(), entry.title, entry.link))
        return news

    @abstractmethod
    def pre_filter(self, entry):
        pass


class HSNewsFetcher(BaseNewsFetcher):
    def __init__(self):
        self.feed_source = "https://www.hs.fi/rss/teasers/etusivu.xml"
        super().__init__()

    # add your filter for this news feed
    def pre_filter(self, entry):
        return recent_news_filter(entry, last_n_hours=12)


class ILNewsFetcher(BaseNewsFetcher):

    def __init__(self):
        self.feed_source = "https://www.iltalehti.fi/rss/uutiset.xml"
        super().__init__()

    def to_standardized_output(self):
        pass

    # add your filter for this news feed
    def pre_filter(self, entry):
        return recent_news_filter(entry, last_n_hours=12)


# Create a list of news fetcher instances
news_fetchers = [ILNewsFetcher(), HSNewsFetcher()]

# Fetch and filter the news
fetched_news = []
for news_fetcher in news_fetchers:
    fetched_news.extend(news_fetcher.get_news())

# Format filtered news as a single string for the API call
news_str = "\n".join(f"{entry[0]} {entry[1]}" for entry in fetched_news)

# Call OpenAI API with the preprompt and filtered news
openai.api_key = api_key
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": preprompt},
        {"role": "user", "content": news_str},
    ]
)

# Get the response from the API
filtered_ids = res["choices"][0]["message"]["content"].split()
filtered_news = [entry for entry in fetched_news if entry[0] in filtered_ids]

# Generate the HTML file
html = '<html>'
for id, title, link in filtered_news:
    html += f"<p style='font-size: 24px;'><a href='{link}'>{title}</a></p>"
html += '</html>'

# Save the HTML file and open it in the browser
path = os.path.abspath('your_ai_powered_news_feed.html')
url = 'file://' + path
with open(path, 'w', encoding='UTF-8') as f:
    f.write(html)

# Open the web page using a local browser
# You may need to adjust this command for your system to automatically open the created page
subprocess.run(f"start chrome {path}", shell=True)
