import openai
import requests
import json
import webbrowser
import tkinter
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
from re import search
import time

class AI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self, system, user):
        messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}
    
    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def next(self, messages: list[dict[str, str]], prompt=None):
        if prompt:
            messages = messages + [{"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            messages=messages,
            stream=True,
            **self.kwargs
        )

        chat = []
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            msg = delta.get('content', '')
            print(msg, end="")
            chat.append(msg)
        return messages + [{"role": "assistant", "content": "".join(chat)}]

    def get_current_weather(self, location: str, unit: str) -> str:
        api_key = "your_api_key"
        res = requests.get(f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no")
        data = json.loads(res.text)
        temp = data["current"]["temp_" + unit]
        desc = data["current"]["condition"]["text"]
        return f"The weather in {location.title()} is {desc.lower()} with a temperature of {temp} {unit.upper()}."

    def search_web(self, query: str) -> str:
        try:
            driver = webdriver.Chrome(ChromeDriverManager().install())
        except:
            return "Error: Webdriver not installed or found. Please install ChromeDriver."
        driver.get("https://www.google.com/")
        search_box = driver.find_element_by_name("q")
        search_box.send_keys(query)
        search_box.submit()
        time.sleep(1)
        results = driver.find_elements_by_css_selector("div.g")
        for result in results:
            link = result.find_element_by_tag_name("a")
            href = link.get_attribute("href")
            if "http" in href:
                parsed_uri = urlparse(href)
                domain = '{uri.netloc}'.format(uri=parsed_uri)
                if search("google|youtube", domain) is None:
                    webbrowser.open_new_tab(href)
                    return "Here's what I found:"
        return "Sorry, I couldn't find anything about that."

    def parse_html(self, html: str, parser="lxml") -> str:
        soup = BeautifulSoup(html, parser)
        return soup.prettify()

    def parse_xml(self, xml: str) -> etree._Element:
        return etree.fromstring(xml)