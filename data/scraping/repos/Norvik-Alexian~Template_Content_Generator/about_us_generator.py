import logging
import os
import config
import openai

from openai.error import APIError
from dotenv import load_dotenv
from utils import prettify

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise EnvironmentError('You should set OPENAI_API_KEY in your environment variable in .env file')


class AboutUsGenerator:
    openai.api_key = OPENAI_API_KEY

    def __init__(self,
                 model=config.MODEL,
                 prompt=config.ABOUT_US_PROMPT,
                 temperature=config.TEMPERATURE,
                 max_tokens=config.MAX_TOKENS,
                 top_p=config.TOP_P,
                 n=config.N,
                 stop=config.STOP,
                 frequency_penalty=config.FREQUENCY_PENALTY,
                 presence_penalty=config.PRESENCE_PENALTY):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def generate_about_us(self, keywords: list):
        """
        :param keywords: list of keywords that we pass to the model to generate about us content
        :return: generated about us content that went through content prettifier
        """
        try:
            model = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are About Us generator that generates the content based on the keywords'},
                    {'role': 'user', 'content': f'{self.prompt}\nKeywords: {keywords}\nAbout Us:'}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=self.n,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            generated_content = model.choices[0].message.content.strip()
            finish_reason = model.choices[0].finish_reason

            return prettify(generated_content, finish_reason)
        except APIError as e:
            message = f'OpenAI model has outage, message: {e}'
            logging.error(message, exc_info=True)
