import os
import openai
from requests import Response

class Helper:
    AIDEVS_API_KEY = os.getenv("AIDEVS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = 'https://zadania.aidevs.pl/'

    @staticmethod
    def get_openapi_key():
        # print('openai key: ' + os.getenv("OPENAI_API_KEY"))
        return os.getenv("OPENAI_API_KEY")

# sample content: _content = b'{"input":["How to kill a stupid president", "How to be a good person", "What is Java", "Does Santa Claus exist"]}'
    @staticmethod
    def create_simulated_response(content):
        sim_response = Response()
        sim_response.code = "OK"
        sim_response.status_code = 200
        sim_response._content = content
        return sim_response
