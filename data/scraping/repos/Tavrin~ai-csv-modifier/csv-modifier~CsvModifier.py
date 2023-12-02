import openai
import pandas as pd
import os
import json
from dotenv import load_dotenv
import math

load_dotenv()

class CsvModifier():
    __engine = 'text-davinci-001'
    __max_tokens = 5
    __temperature = 0.7
    __test = 'false'
    __api_key = None
    __top_p = 1
    __prompt = None
    __post_item_prompt = ''
    sdk_config = {}
    __n = 1
    __stream = False
    __csv_file = '../data.csv'
    __file_name_format = 'enumerate'
    __file_path = '../output'
    __column_to_read = None
    __column_to_modify = None
    __values_to_ignore = ['#VALUE!']
    responses_to_modify = {
        'feminine': [
            'Féminine',
            'féminine',
            'Feminine',
            'feminine',
            'Féminin',
        ],
        'masculine': [
            'Masculine',
            'masculine',
            'Masculin',
        ]
    }

    def __init__(self, config):
        self.setProperties(config)

    def setProperties(self, config):
        if (os.getenv('API_KEY') is None):
            raise ValueError('No API KEY added to .env file')

        self.__api_key = os.getenv('API_KEY')

        if ('column_to_read' in config):
            self.__column_to_read = config['column_to_read']

        if ('column_to_modify' in config):
            self.__column_to_modify = config['column_to_modify']

        if ('csv_file' in config):
            self.__csv_file = os.getcwd() + config['csv_file']

        if ('prompt' in config):
            self.__prompt = config['prompt']

        if ('post_item_prompt' in config):
            self.__post_item_prompt = config['post_item_prompt']

        if ('test' in config):
            self.__test = config['test']

    def generate(self):
        data = pd.read_csv(self.__csv_file, encoding='utf8')

        for index, column in enumerate(data[self.__column_to_read]):
            if (column in self.__values_to_ignore) or isinstance(column, str) == False or (self.__test == 'true' and index > 50):
                continue

            column = column.split()[0]
            prompt = self.__prompt + ' ' + column + ' ' + self.__post_item_prompt
            response = self.callApi(prompt)

            print('[' + str(index) + '] ' + column + ' : ' + response)
            data.loc[index, self.__column_to_modify] = response

        data.to_csv('newdata.csv')

    def callApi(self, prompt):
        openai.api_key = self.__api_key
        completion = openai.Completion.create(
            engine = self.__engine,
            prompt = prompt,
            max_tokens = self.__max_tokens
        )

        return self.sanitizeResponse(completion.choices[0].text)

    def sanitizeResponse(self, response):
        response = response.split()[0]
        data_to_return = ''.join(ch for ch in response if ch.isalnum())

        if data_to_return in self.responses_to_modify['feminine']:
            data_to_return = 'féminin'

        if data_to_return in self.responses_to_modify['masculine']:
            data_to_return = 'masculin'

        return data_to_return