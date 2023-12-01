import feedparser
from config import load_config
from gpt import generate_content, generate_title
import openai
import os
import json
from datetime import datetime

config = load_config()

with open(config['folders']['prompts'] + '/system_content.txt') as f:
    system_content_prompt = f.read()

with open(config['folders']['prompts'] + '/user_content.txt') as f:
    user_content_prompt = f.read()  

def load_seen_links():
    seen_links = set()
    file_path = "seen_links.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            if isinstance(data, dict):
                for date, links in data.items():
                    for link in links:
                        seen_links.add(link)

    return seen_links

def save_seen_links(seen_links):
    file_path = "seen_links.json"
    data = {}

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

    current_date = datetime.now().strftime("%Y-%m-%d")
    if current_date not in data:
        data[current_date] = []

    data[current_date].extend(seen_links)

    with open(file_path, "w") as file:
        json.dump(data, file)

def check_feed(feed_url, seen_links, num_articles):
    feed = feedparser.parse(feed_url)

    for item in feed.entries:
        if item.link not in seen_links:
            # Generate content and extract text
            content = generate_content(item.description, item.link)
            if isinstance(content, openai.Completion):
                content = content.choices[0].text

            # Generate title from content text
            title = generate_title(content)

            seen_links.add(item.link)
            num_articles += 1

            return title, content

    return None
