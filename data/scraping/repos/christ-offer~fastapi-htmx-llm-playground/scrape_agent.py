import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Union
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os

class Scraper:
    def __init__(
            self, 
            model: str = "gpt-4-0613", 
            temperature: float = 0.1, 
            top_p: float = 1.0, 
            frequency_penalty: float = 0.0, 
            presence_penalty: float = 0.0
            ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.function_params = self.scrape_params
        load_dotenv()
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.system_message = """# Scrape Webpage Agent
This agent scrapes a webpage and returns the result as a JSON string.
"""
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def top_p(self):
        return self._top_p

    @top_p.setter
    def top_p(self, value):
        self._top_p = value

    @property
    def frequency_penalty(self):
        return self._frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value):
        self._frequency_penalty = value

    @property
    def presence_penalty(self):
        return self._presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value):
        self._presence_penalty = value
        
    @property
    def system_message(self) -> str:
        return self._system_message
    
    @system_message.setter
    def system_message(self, value):
        self._system_message = value

    def scrape_webpage(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

            # Parse the webpage with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Define the tags to be removed
            tags = ['a', 'img', 'script', 'style', 'svg', 'iframe', 'canvas', 'video', 'audio', 'map', 'noscript']

            # Remove each defined tag
            for tag in tags:
                for item in soup.find_all(tag):
                    item.decompose()

            # Extract text from the parsed HTML
            text = soup.get_text()

            # Remove extra whitespace
            text = ' '.join(text.split())
            completion = self.anthropic.completions.create(
                model="claude-2",
                max_tokens_to_sample=1000000,
                prompt=f"{HUMAN_PROMPT} Please provide a exhaustive and concise summary of the following (formatted nicely in markdown): {text} {AI_PROMPT}",
                stream=False,
            )
            return completion.completion
        except requests.HTTPError as e:
            return f"A HTTP error occurred: {str(e)}"
        except requests.RequestException as e:
            return f"A request exception occurred: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    @property
    def scrape_params(self) -> List[Dict[str, Union[str, Dict]]]:
        return [
            {
                "name": "scrape_webpage",
                "description": "Scrapes a webpage and returns the result as a JSON string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the webpage to scrape."},
                    },
                    "required": ["url"],
                },
            }
        ]
