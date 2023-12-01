from typing import List, Dict
from model.model import Model
import os
from time import sleep
import requests
import anthropic
if os.name == 'nt':
    os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/anthropic.crt"
class AnthropicModel(Model):

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.token = os.environ['ANTHROPIC_API_KEY']
    def call_model_rest(self, prompt):
        '''
        Will call model with rest api
        :return:
        '''
        if os.name == 'nt':
            os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/anthropic.crt"

        headers = {
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
            'accept': 'application/json',
            'x-api-key': self.token
        }

        data = {
            "prompt": prompt,
            "model": self.cfg['model_name'],
            "max_tokens_to_sample": self.cfg['max_output_tokens'],

        }
        try:
            response = requests.post(
                url="https://api.anthropic.com/v1/complete",
                json=data,
                headers=headers,
                verify=False
            )
            if response.status_code != 200:
                print(response.json())
                raise ValueError("Failed to call model")

            res = response.json()
        except Exception as e:
            print('FAILED TO CALL MODEL, trying again')
            print(e)
            sleep(3)
            response = requests.post(
                url="https://api.anthropic.com/v1/complete",
                json=data,
                headers=headers,
                verify=False
            )
            res = response.json()

        if 'error' in res:
            raise ValueError(f"Error calling model {self.cfg['model_name']} with prompt {prompt} : {res['error']}")
        print(response.json())
        return response.json()['completion']

    def generate(self, prompts: List[str], temp:float = 0.0):
        '''
        Will generate content from given prompt

        :param prompts:
        :param temp:
        :return:
        '''

        prompts = [f"\n\n{anthropic.HUMAN_PROMPT} {p}{anthropic.AI_PROMPT}" if "HUMAN" not in p else p for p in prompts]
        responses = []
        for prompt in prompts:
            answer = self.call_model_rest(prompt)
            responses.append(answer)
        return responses