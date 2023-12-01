import openai
import os
import json
from dotenv import load_dotenv
from google_search import *
from gpt_functions import *
import pprint

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class GoogleGPT:

    def __init__(self):

        self.links = []
        self.titles = []
        self.base_query = None
        self.query = None
        self.summaries = []

    def search(self, query):
        self.base_query = query
        payload = craft_payload(query)
        self.query = payload['q']
        pprint.pprint(payload)
        titles_and_links = search_results(payload=payload)
        selected_link = decide_leads(titles_and_links, self.query)['link']
        self.links.append(selected_link)
        self.titles.append(list(titles_and_links.keys())[list(
            titles_and_links.values()).index(selected_link)])

        for link in self.links:
            content = extract_webpage_content_efficient(link)
            summary = summarize(self.base_query, content)
            self.summaries.append(summary['summary'])
        
        pprint.pprint(self.summaries)

gGPT = GoogleGPT()
gGPT.search('how many people have died in palestine from the latest conflict')
