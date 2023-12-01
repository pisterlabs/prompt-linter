import os

from dotenv import load_dotenv
from quantulum3 import parser

from closer.legal_openai.openai_tasks import OpenaiTask

load_dotenv()
prompt_path = os.getenv('PROMPT_PATH')


class QuantitiesExtractor:
    def __init__(self):
        self.prompt_path = prompt_path

    def quantulum_extract(self, text):
        quants = parser.parse(text)
        temp_list = []
        for quantity in quants:
            if quantity.unit.name == 'dimensionless':
                continue
            if quantity.surface is not None:
                temp_list.append(quantity.surface)
        return temp_list

    def openai_extract(self, prompt=None, article=None, path=None, api_key=None):
        if prompt is None:
            with open(self.prompt_path + 'quantity.txt', 'r') as f:
                prompt = f.read()
        return OpenaiTask(path=path, api_key=api_key).execute_task(
            prompt=prompt, article=article)
