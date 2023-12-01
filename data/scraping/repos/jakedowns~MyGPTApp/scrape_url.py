# a class that can scrape a url and return the text (stripped of html tags)
# and the title of the page
import requests
from bs4 import BeautifulSoup
from mygptapp import rules, db
from mygptapp.models import Message
from mygptapp.openai_api import OpenAIAPI
from mygptapp.utils import save_and_emit_message

from mygptapp.actions.pyppeteer_scraper import PyppeteerScraper

ai_api = OpenAIAPI()

#DEFAULT_MODE = "pyppeteer" #broken
DEFAULT_MODE = "requests"
class ScrapeURL:
    def __init__(self, url, mode = DEFAULT_MODE):
        self.url = url
        self.mode = mode
        self.title = None
        self.text = None

    async def scrape(self, url):
        if self.mode == "pyppeteer":
            return await self.scrape_pyppeteer(url)
        #elif self.mode == "requests":
        else:
            return self.scrape_requests(url)

    def scrape_pyppeteer_fg(self, url):
        # scrape the url and set the title and text attributes
        # use the pyppeteer library
        scraper = PyppeteerScraper()
        scraper.init_sync()
        html = scraper.scrape_sync(url)
        soup = BeautifulSoup(html, "html.parser")
        # if title is found, set it
        if soup.title and "string" in soup.title:
            self.title = soup.title.string
        self.text = soup.get_text()

    async def scrape_pyppeteer(self, url):
        # scrape the url and set the title and text attributes
        # use the pyppeteer library
        scraper = PyppeteerScraper()
        html = await scraper.scrape_in_background(url)
        soup = BeautifulSoup(html, "html.parser")
        # if title is found, set it
        if soup.title and "string" in soup.title:
            self.title = soup.title.string
        self.text = soup.get_text()

    def scrape_requests(self,url):
        # scrape the url and set the title and text attributes
        # use the requests and BeautifulSoup libraries
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        # if title is found, set it
        if soup.title and "string" in soup.title:
            self.title = soup.title.string
        self.text = soup.get_text()

    def get_response(self):
        return {
            "title": self.title,
            "text": self.text
        }

    def process_scrape_results(self, original_prompt):
        # convert the self object to a message
        scrape_results_as_message = "Scrape Results for Url: \n```" + self.url
        if self.title is not None:
            scrape_results_as_message += "\n"
            scrape_results_as_message += "title: " + self.title
        if self.text is not None:
            scrape_results_as_message += "\n"
            # cap the body text at N characters
            scrape_results_as_message += "body text: " + self.text[:3000]
        scrape_results_as_message += "\n```\n"

        # save the scrape message to the db
        # scrape_message = Message(convo_id=1, user_id=1, content=scrape_results_as_message, is_inner_thought=False, role="system")
        # db.session.add(scrape_message)
        # db.session.commit()

        # store the message to the db and emit it to the client
        save_and_emit_message(
            convo_id=1,
            user_id=1,
            role="system",
            content=scrape_results_as_message,
        )

        # bake scrape_results into a "chat"
        messages = []
        messages.append({"role": "user", "content": rules.get_preamble_text()})
        messages.append({"role": "user", "content": original_prompt})
        messages.append({"role": "system", "content": scrape_results_as_message})
        messages.append({"role": "system", "content": "See if you can respond to the base prompt now using the scrape results using a final_response action, or another follow-up action if required"})

        print("calling the model with a followup prompt: ", messages)
        followup = ai_api.call_model(messages)
        followup["scrape_result_message"] = {
            "choices":[
                {
                    "message": {
                        "role": "system",
                        "content": scrape_results_as_message
                    }
                }
            ]
        }
        return followup